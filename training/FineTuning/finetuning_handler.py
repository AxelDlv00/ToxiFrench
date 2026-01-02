""" 
# ╔════════════════════════════════════════════════════════╗
# ║           UTILITY CLASS FOR QLORA FINETUNING           ║
# ╚════════════════════════════════════════════════════════╝
"""

# ╔════════════════════════════════════════════════════════╗
# ║                       Libraries                        ║
# ╚════════════════════════════════════════════════════════╝

import os
import sys
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import argparse

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, PeftModel
from trl import SFTConfig, DataCollatorForCompletionOnlyLM, SFTTrainer
import pytorch_optimizer as optim
from accelerate import PartialState 

from utils.dataset_handling import DatasetPreparer
from utils.dynamic_weigthed_loss import CustomDWLTrainer

# ╔════════════════════════════════════════════════════════╗
# ║                   QLoRAModelHandler                    ║
# ╚════════════════════════════════════════════════════════╝

class QLoRAModelHandler:
    def __init__(self, **kwargs):
        self.console = Console()
        self._show_banner()
        
        # Attributes
        self.model_name = kwargs.get("model_name", "Qwen/Qwen3-4B")
        self.local_checkpoint = kwargs.get("local_checkpoint")
        self.mode = kwargs.get("mode", "train")
        self.device = self._setup_device()

        # Dataset fields configuration
        self.dataset_path = kwargs.get("dataset_path", "Naela00/ToxiFrench")
        self.text_field = kwargs.get("text_field", "content")
        self.cot_fields = kwargs.get("cot_fields", ["direct_question"])
        self.label_field = kwargs.get("label_field", "literal_conclusion_annotator")
        self.optimizer_type = kwargs.get("optimizer_type", "AdamW")
        self.proxy = kwargs.get("proxy_address", "")
        
        # BitsAndBytes configuration
        self._set_proxy()
        self.bnb_config = self._get_bnb_config(kwargs.get("bnb_config_kwargs"))
        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model()
        
        # Post-initialization steps
        self._sync_vocabulary()
        self._load_adapters_if_exists()
        self._print_model_stats()

    # +--------------------------------------------------------+
    # |                    Private Methods                     |
    # +--------------------------------------------------------+

    def _set_proxy(self):
        if self.proxy:
            os.environ["HTTP_PROXY"] = self.proxy
            os.environ["HTTPS_PROXY"] = self.proxy

    def _show_banner(self):
        # Only print banner on main process to avoid log spam
        if PartialState().is_main_process:
            banner_path = Path('assets/banner.txt')
            content = banner_path.read_text(encoding='utf-8') if banner_path.exists() else "QLoRA Handler"
            self.console.print(Panel.fit(Text(content, style="bold green"), title="QLoRA Utility", style="cyan"))

    def _setup_device(self) -> str:
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            return f"cuda:{local_rank}"
        return "cpu"

    def _get_bnb_config(self, custom_kwargs: Optional[dict]) -> BitsAndBytesConfig:
        default = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16
        }
        config = custom_kwargs or default
        return BitsAndBytesConfig(**config)

    def _init_tokenizer(self):
        path = self.local_checkpoint or self.model_name
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        tokens = ["<think>", "</think>"]
        tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        return tokenizer

    def _init_model(self):
        # We only show the spinner on the main process
        if PartialState().is_main_process:
            context = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console) 
        else:
            # Dummy context manager for other processes
            import contextlib
            context = contextlib.nullcontext()

        with context as progress:
            if progress: progress.add_task(f"Loading {self.model_name}...", total=None)
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map={"": self.device} if self.device != "cpu" else "auto",
                quantization_config=self.bnb_config if self.device != "cpu" else None,
                trust_remote_code=True,
                sliding_window=None
            )

            model.config.use_cache = True if self.mode != "train" else False

            if model.generation_config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
                model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        return model

    def _sync_vocabulary(self):
        """
        Dynamically resizes the model's embeddings to match the tokenizer's vocabulary size.
        """
        tokenizer_vocab_size = len(self.tokenizer)
        model_embedding_size = self.model.get_input_embeddings().weight.size(0)

        if model_embedding_size != tokenizer_vocab_size:
            if PartialState().is_main_process:
                self.console.print(f"[yellow]Vocab mismatch detected: Model={model_embedding_size}, Tokenizer={tokenizer_vocab_size}[/yellow]")
                self.console.print(f"[bold cyan]Resizing model embeddings to match tokenizer ({tokenizer_vocab_size})...[/bold cyan]")
            
            self.model.resize_token_embeddings(tokenizer_vocab_size)
            
            if PartialState().is_main_process:
                new_size = self.model.get_input_embeddings().weight.size(0)
                self.console.print(f"[green]New model embedding size: {new_size}[/green]")
        elif PartialState().is_main_process:
            self.console.print("[green]Vocabulary sizes already match.[/green]")

    def _load_adapters_if_exists(self):
        if self.local_checkpoint and self.mode != "train":
            if PartialState().is_main_process:
                self.console.print(f"[yellow]Loading adapters from {self.local_checkpoint}...[/yellow]")
            self.model = PeftModel.from_pretrained( 
                self.model, 
                self.local_checkpoint, 
                is_trainable=False
            ).to(self.device)

    def _print_model_stats(self):
        if PartialState().is_main_process:
            max_len = getattr(self.model.config, "max_position_embeddings", "Unknown")
            self.console.print(f"[bold cyan]Model Ready.[/bold cyan] Max Context: {max_len} tokens.")

    # +--------------------------------------------------------+
    # |                    Public Methods                      |
    # +--------------------------------------------------------+

    def train(self, peft_kwargs: dict = None, training_kwargs: dict = None, custom_kwargs: dict = None):

        if PartialState().is_main_process:
            self.console.print("[cyan]Preparing dataset...[/cyan]")

        preparer = DatasetPreparer(
            self.tokenizer, 
            text_field=self.text_field, 
            cot_fields=self.cot_fields, 
            label_field=self.label_field,
            seed=custom_kwargs.get("dataset", {}).get("seed", 42),
            oversampling=custom_kwargs.get("dataset", {}).get("oversampling", False),
            max_length=training_kwargs.get("max_length", 1024)
        )
        
        with PartialState().main_process_first():
            train_ds = preparer.prepare(self.dataset_path, split_name=custom_kwargs.get("dataset", {}).get("train_split_name", "train"))
            eval_ds = preparer.prepare(self.dataset_path, split_name=custom_kwargs.get("dataset", {}).get("eval_split_name", "test"))

        if PartialState().is_main_process:
            print(f"Train samples: {len(train_ds)} | Eval samples: {len(eval_ds)}")

        peft_config = self._build_lora_config(peft_kwargs)
        sft_config = self._build_sft_config(training_kwargs)
        optimizer = self._setup_optimizer(sft_config.learning_rate, sft_config.weight_decay)

        custom_kwargs = custom_kwargs or {}
        self.weight_schedule = custom_kwargs.get("weight_schedule", [
            {"epoch": 0, "alphas": [1.0] * (len(self.cot_fields) + 1)}
        ])
        
        # Normalization commented out to match Code 1
        # for entry in self.weight_schedule:
        #     total = sum(entry["alphas"])
        #     entry["alphas"] = [a / total for a in entry["alphas"]]

        response_template = "Analyse:\n"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template, 
            tokenizer=self.tokenizer
        )

        self.trainer = CustomDWLTrainer(
            model=self.model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            args=sft_config,
            peft_config=peft_config,
            processing_class=self.tokenizer,
            optimizers=(optimizer, None),
            think_start_id=self.tokenizer.convert_tokens_to_ids("<think>"),
            think_end_id=self.tokenizer.convert_tokens_to_ids("</think>"),
            weight_schedule=self.weight_schedule,
            DFT=custom_kwargs.get("DFT", False)
        )

        if PartialState().is_main_process:
            self.console.rule("[bold cyan]Starting Training[/bold cyan]")
        
        self.trainer.train(resume_from_checkpoint=self.local_checkpoint)
        
        self._save_final_model(sft_config.output_dir)

    # +--------------------------------------------------------+
    # |                    Internal Helpers                    |
    # +--------------------------------------------------------+

    def _build_lora_config(self, kwargs) -> LoraConfig:
        kwargs = kwargs or {}
        config_params = {
            "task_type": TaskType.CAUSAL_LM,
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
            "bias": "none"
        }
        config_params.update(kwargs)
        return LoraConfig(**config_params)

    def _build_sft_config(self, kwargs) -> SFTConfig:
        default_args = {
            "output_dir": "./output_qlora",
            "max_length": 1024,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 16,
            "learning_rate": 2e-4,
            "logging_steps": 10,
        }
        if kwargs: default_args.update(kwargs)
        return SFTConfig(**default_args)

    def _save_final_model(self, output_dir):
        # SFTTrainer handles saving on main process automatically, but for safety:
        if PartialState().is_main_process:
            path = Path(output_dir) / "final_adapters"
            self.trainer.save_model(path)
            self.console.print(f"[bold green]Model saved: {path}[/bold green]")

    def _setup_optimizer(self, lr: float, wd: float):
        """Initializes the optimizer based on the specified type."""
        if hasattr(self, "optimizer_type") and self.optimizer_type.lower() == 'soap':
            try:
                if PartialState().is_main_process:
                    self.console.print("[bold cyan]Using SOAP Optimizer[/bold cyan]")
                return optim.SOAP(
                    self.model.parameters(),
                    lr=lr,
                    betas=(0.9, 0.99),
                    weight_decay=wd,
                    precondition_frequency=10
                )
            except ImportError:
                if PartialState().is_main_process:
                    self.console.print("[red]pytorch_optimizer not found. Falling back to AdamW.[/red]")
        
        if PartialState().is_main_process:
            self.console.print("[bold cyan]Using AdamW Optimizer (Default)[/bold cyan]")
        return None 

    # +--------------------------------------------------------+
    # |                   Inference Methods                    |
    # +--------------------------------------------------------+

    def generate_text(self, prompt: str, **kwargs):
        """Generates a response from the model for a given prompt."""
        if not self.model:
            self.console.print("[bold red]Model not loaded![/bold red]")
            return
        
        gen_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("do_sample", True),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)
            
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        return full_text

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        if not self.model:
            return []
        
        gen_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "do_sample": kwargs.get("do_sample", True),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)
            
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        return decoded_outputs

def main():
    epilog = """
Example of command line usage (Multi-GPU/DeepSpeed) :
  accelerate launch launch_dpo.py --config config.json

Simple example (Single GPU) :
  python main.py --config config.json
    """

    parser = argparse.ArgumentParser(
        description="QLoRA Training Launcher",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter 
    )

    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json", 
        help="Path to the JSON configuration file (default: config.json)"
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        Console().print(f"[bold red]Error:[/bold red] Configuration file [yellow]{args.config}[/yellow] not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    bnb_kwargs = config.get("bnb_params", {})
    if bnb_kwargs.get("bnb_4bit_compute_dtype") == "float16":
        bnb_kwargs["bnb_4bit_compute_dtype"] = torch.float16
    elif bnb_kwargs.get("bnb_4bit_compute_dtype") == "bfloat16":
        bnb_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16

    handler = QLoRAModelHandler(
        **config.get("model_params", {}),
        **config.get("dataset_params", {}),
        bnb_config_kwargs=bnb_kwargs
    )

    mode = config.get("model_params", {}).get("mode", "train")
    if mode == "train":
        handler.train(
            peft_kwargs=config.get("lora_params"),
            training_kwargs=config.get("training_params"),
            custom_kwargs=config.get("training_params_custom")
        )
    else:
        preparer = DatasetPreparer(
            handler.tokenizer, 
            handler.text_field, 
            handler.cot_fields, 
            handler.label_field
        )
        # For testing dataset generation, we also need the main_process_first block
        with PartialState().main_process_first():
            ds_test = preparer.prepare(handler.dataset_path, split_name="test")
            
        for i in range(1):
            sample = ds_test[i]
            full_decoded = handler.tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
            prompt = full_decoded.split("Analyse:")[0] + "Analyse:\n<think>\n"
            expected = full_decoded.split("Analyse:")[-1]

            Console().print(Panel(expected, title=f"Expected Output {i+1}", border_style="magenta"))
            Console().print(Panel(prompt, title=f"Prompt Sample {i+1}", border_style="blue"))
            
            output = handler.generate_text(prompt)
            Console().print(Panel(output, title=f"Generated Output {i+1}", border_style="green"))

if __name__ == "__main__":
    main()