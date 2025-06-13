""" 

# ╔════════════════════════════════════════════════════════╗
# ║           UTILITY CLASS FOR QLORA FINETUNING           ║
# ╚════════════════════════════════════════════════════════╝

This script provides a utility class `QLoRAModelHandler` for loading a pre-trained model,
performing inference, and fine-tuning using QLoRA (Quantized Low-Rank Adaptation) with the Hugging Face Transformers library.
It supports loading models in 4-bit quantization, preparing datasets, and training with LoRA adapters.
It also includes methods for generating text and preparing datasets for training.

# Note : If you want to monitor the training progress while in remote server, you can use TensorBoard like this :
    - On the remote server, run: `tensorboard --logdir runs/exp1`
    - On the local machine, run something like:
        `ssh -L 6006:localhost:6006 user@remote_server` (e.g. `ssh -L 6006:localhost:6006 SJTU`)
    - Open your browser and go to `http://localhost:6006` to view the TensorBoard dashboard.

# Note : If you want to launch the process you can use `tmux` : 
    - Start a new tmux session: `tmux new -s qlora_session`
    - Run your script: `accelerate launch qlora_handler.py --mode train --dataset_name rdal`
    - Detach from the session: Press `Ctrl + b`, then `d`
    - Reattach to the session later: `tmux attach -t qlora_session`
    - To kill the session: `tmux kill-session -t qlora_session` if you are outside the session, or `exit` if you are inside the session.
"""

# ╔════════════════════════════════════════════════════════╗
# ║                       Libraries                        ║
# ╚════════════════════════════════════════════════════════╝

import os
import json
import warnings
from pathlib import Path
from functools import partial
from typing import Optional, Dict, Any
import argparse
import re

import torch
from datasets import load_dataset, Dataset
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
import sys
import torch.distributed as dist

# ╔════════════════════════════════════════════════════════╗
# ║        Logging handling in distributed training        ║
# ╚════════════════════════════════════════════════════════╝

def is_rank_0_env():
    rank = os.environ.get("RANK", "0")
    print(f"Running on RANK {rank}")
    return int(rank) == 0

if not is_rank_0_env():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

# ╔════════════════════════════════════════════════════════╗
# ║                         Banner                         ║
# ╚════════════════════════════════════════════════════════╝

# Using `shadow` font for the banner
banner = Path('utils/banner.txt').read_text(encoding='utf-8')

# ╔════════════════════════════════════════════════════════╗
# ║                        Utility                         ║
# ╚════════════════════════════════════════════════════════╝

mapping_thouhts = {
    "<cot_explication>": "Explication :",
    "<cot_ton>": "Tons :",
    "<cot_intention>": "Intentions :",
    "<cot_categorie_list>": "Catégorie(s) de toxicité implicite :",
    "<cot_categorie_justification>": "Justification :",
    "<cot_labels_list>": "Labels :",
    "<cot_labels_justification>": "Justification :",
    "<cot_toxicite_note>": "Score de toxicité :",
    "<cot_toxicite_justification>": "Justification :",
}

def remove_thought(text: str, thought_token: str = "<cot_categorie_list>") -> str:
    escaped_token = re.escape(mapping_thouhts[thought_token])
    pattern = rf"<think>\s*{escaped_token}.*?</think>\n?"
    return re.sub(pattern, "", text, flags=re.DOTALL)

def formatting_func(example, text_field, cot_field, label_field, tokenizer, remove_thoughts):
    prompt = f"Message:\n{example[text_field].strip()}\n\nAnalyse:\n"
    thought = example[cot_field].strip()
    explication_pattern = re.compile(r"(<think>\s*Explication :\s*\n?)", re.DOTALL)
    match = explication_pattern.search(thought)
    if match:
        explication_block = match.group(1)
        prompt += explication_block
        thought = thought[match.end():].strip()
    for thought_token in remove_thoughts:
        thought = remove_thought(thought, thought_token)
    completion = (
        f"{thought}\n\n"
        # f"Conclusion:\n"
        f"{example[label_field].strip()}"
        f"{tokenizer.eos_token}"
    )
    return {"prompt": prompt, "completion": completion}

# ╔════════════════════════════════════════════════════════╗
# ║                   QLoRAModelHandler                    ║
# ╚════════════════════════════════════════════════════════╝

class QLoRAModelHandler:
    def __init__(self,
                 ### Model Loading Parameters
                 model_name:                    str = "Qwen/Qwen3-4B", # "Qwen/Qwen2.5-3B", # "Qwen/Qwen3-4B-Base",
                 proxy_address:                 str = "socks5h://127.0.0.1:1080",
                 use_proxy:                     str = False,
                 bnb_config_kwargs:             dict = None,
                 default_lora_target_modules:   list = None,
                 local_checkpoint:              str = None,             # Path to a local checkpoint to load the model from
                 ### Dataset Preparation Parameters
                 dataset_path:  str = "Naela00/ToxiFrenchFinetuning",   # Link to the dataset on Hugging Face
                 dataset_name:  str = "odal",                           # Name of the dataset to load
                 remove_thoughts: list[str] = ["<cot_intention>", "<cot_categorie_list>", "<cot_labels_list>"], # Comma-separated list of thought tokens to remove from the generated text
                 ### Fields for Dataset Preparation
                 text_field:    str = "content",                        # Field in the dataset to use for training
                 cot_field:     str = "cot_text",                       # Field for chain-of-thought reasoning, if applicable
                 label_field:   str = "literal_conclusion_annotator"    # Field for labels, if applicable
                 ):
        
        ### Logging and Console Setup
        self.console = Console()
        self.console.print('\n')
        self.console.print(Panel.fit(Text(banner, style="bold green"), title="QLoRA Utility", style="cyan"))
        self.console.print('\n')
        self.console.print(Markdown(
            f"* Using model: `{model_name}`\n"
            f"* Dataset: `{dataset_path}` (`{dataset_name}`)\n"
            f"* text / cot / label fields: "
            f"`{text_field}` | `{cot_field}` | `{label_field}`"))
        self.console.print()
        self.console.rule(f"[bold cyan]Initializing handler for {model_name}")
        self.console.print()

        ### Model Loading Parameters
        self.model_name:    str = model_name
        self.proxy_address: str = proxy_address
        self.tokenizer:     AutoTokenizer = None
        self.model:         AutoModelForCausalLM = None
        self.device:        str = None
        self.local_checkpoint: str = local_checkpoint 

        ### Trainer and Dataset Parameters
        self.trainer:       SFTTrainer = None
        self.text_field:    str = text_field
        self.cot_field:     str = cot_field
        self.label_field:   str = label_field
        self.dataset_path:  str = dataset_path
        self.dataset_name:  str = dataset_name
        self.dataset:       Dataset = None
        self.remove_thoughts: list[str] = remove_thoughts

        ### Proxy Setup
        if use_proxy:
            self._setup_proxy()

        ### Default BitsAndBytes config
        self.bnb_config_kwargs = bnb_config_kwargs if bnb_config_kwargs is not None else {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16
        }
        self.bnb_config = self._get_bnb_config(**self.bnb_config_kwargs)

        ### Default LoRA target modules (Qwen specific, adjust if using other models)
        self.default_lora_target_modules = default_lora_target_modules if default_lora_target_modules is not None else [
            "q_proj", "k_proj", "v_proj", "o_proj", # https://github.com/axolotl-ai-cloud/axolotl/blob/v0.9.2/examples/qwen3/32b-qlora.yaml
            "up_proj", "down_proj"
        ]

        ### Load Tokenizer and Model
        self.console.print(f"Using model: {self.model_name}")
        self.load_tokenizer_and_model()
        self.console.print()

        # Checking the window size of the model
        self.console.print('\n')
        self.console.rule("[bold yellow]Model Configuration")
        max_len = self.model.config.max_position_embeddings
        window_size = getattr(self.model.config, 'sliding_window', 'N/A') # Check for sliding window
        self.console.print(f"Model Max Sequence Length: [bold cyan]{max_len}[/bold cyan] tokens")
        self.console.print(f"Sliding Window Attention Size: [bold cyan]{window_size}[/bold cyan] tokens")
        self.console.print()

    # ╭────────────────────────────────────────────────────────╮
    # │                     Model handling                     │
    # ╰────────────────────────────────────────────────────────╯

    ### Sets up the proxy environment variables if a proxy is used.
    def _setup_proxy(self) -> None:
        """Sets up proxy environment variables."""
        os.environ["HTTP_PROXY"] = self.proxy_address
        os.environ["HTTPS_PROXY"] = self.proxy_address
        self.console.print(f"Proxy configured to: {self.proxy_address}")

    ### Creates and returns a BitsAndBytesConfig based on the provided kwargs.
    def _get_bnb_config(self, **kwargs) -> BitsAndBytesConfig:
        """Creates and returns a BitsAndBytesConfig."""
        self.console.print(f"Using BitsAndBytesConfig: {kwargs}")
        return BitsAndBytesConfig(**kwargs)

    ### Loads the tokenizer and the model, applying 4-bit quantization if available.
    def load_tokenizer_and_model(self) -> None:
        """Loads the tokenizer and the 4-bit quantized model."""
        self.console.print('\n')
        self.console.rule("[bold cyan]Loading Tokenizer and Model", style="cyan")
        self.console.print('\n')

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >      Handling CUDA availability and device setup       >
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
            self.console.print(f"Process {local_rank} using device: {torch.cuda.get_device_name(local_rank)}")
        else:
            self.device = "cpu"
            self.console.print("CUDA not available. Using CPU (Note: Performance will be significantly slower).")
            if self.bnb_config.load_in_4bit or self.bnb_config.load_in_8bit:
                self.console.print("[yellow]Warning: BitsAndBytes quantization is primarily for GPUs. CPU performance may be poor or unsupported.[/yellow]")
        self.console.print()

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >                   Load the Tokenizer                   >
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # tokenizer_path = self.local_checkpoint if self.local_checkpoint else self.model_name
        tokenizer_path = self.model_name
        base_model_path = self.model_name  # The base model is always from the original path

        self.console.print(f"Loading tokenizer from: [bold cyan]{tokenizer_path}[/bold cyan]")


        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console) as progress:
            task_tokenizer = progress.add_task("Loading tokenizer...", total=None)
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=True,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.console.print("Tokenizer `pad_token` was None, set to `eos_token`.")
            self.tokenizer.padding_side = 'left' # FlashAttention2 requires left padding
            self.console.print(f"[bold green]Tokenizer '{self.tokenizer.name_or_path}' loaded successfully![/bold green]")
            self.console.print('\n')
            progress.update(task_tokenizer, completed=1, description="Tokenizer loaded.")
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >                     Load the model                     >
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console) as progress:
            task_model = progress.add_task("Loading model...", total=None)
            device_map_config = {"": self.device} if torch.cuda.is_available() else "auto" # For CPU, device_map should not be set to current_device if it's a GPU index

            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map=device_map_config,
                quantization_config=self.bnb_config if torch.cuda.is_available() else None, # Only apply quantization if on CUDA
                trust_remote_code=True,
                sliding_window=None, # Or a specific value if known and desired
                # attn_implementation= # "flash_attention_2",
            )
            if self.model.generation_config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
                self.console.print("Model `generation_config.pad_token_id` set from tokenizer.")
                
            self.console.print(f"[bold green]Model '{self.model_name}' loaded successfully![/bold green]")
            self.console.print('\n')
            progress.update(task_model, completed=1, description="Model loaded.")
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >                Special Tokens Handling                 >
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.console.rule("[bold yellow]Handling Special Tokens & Resizing")

        # Define the special tokens to ensure are present
        special_tokens_to_add = {
            "additional_special_tokens": [
                # "<cot_explication>", "<cot_ton>", "<cot_intention>",
                # "<cot_categorie_list>", "<cot_categorie_justification>",
                # "<cot_labels_list>", "<cot_labels_justification>",
                # "<cot_toxicite_note>", "<cot_toxicite_justification>",
            ]
        } if "Qwen3" in self.model_name else {
            "additional_special_tokens": [
                "<think>", "</think>", 
                # "<cot_explication>", "<cot_ton>",
                # "<cot_intention>", "<cot_categorie_list>",
                # "<cot_categorie_justification>", "<cot_labels_list>",
                # "<cot_labels_justification>", "<cot_toxicite_note>",
                # "<cot_toxicite_justification>"
            ]
        }
        
        # This call ensures tokens are added if they don't exist (for training)
        # and does nothing if they already exist (for inference from checkpoint).
        self.tokenizer.add_special_tokens(special_tokens_to_add)

        # --- The CRITICAL resizing logic ---
        # Check if the model's embedding layer size matches the tokenizer's vocabulary size.
        model_embedding_size = self.model.get_input_embeddings().weight.size(0)
        tokenizer_vocab_size = len(self.tokenizer)

        if model_embedding_size != tokenizer_vocab_size:
            self.console.print(f"[yellow]Warning: Vocab size mismatch.[/yellow] "
                               f"Model embeddings: {model_embedding_size}, "
                               f"Tokenizer vocab: {tokenizer_vocab_size}")
            self.console.print("Resizing model token embeddings to match tokenizer...")
            self.model.resize_token_embeddings(tokenizer_vocab_size)
            
            # Verify the resize
            new_embedding_size = self.model.get_input_embeddings().weight.size(0)
            self.console.print(f"[bold green]Resized model embeddings to: {new_embedding_size}[/bold green]")
        else:
            self.console.print("Model embedding size and tokenizer vocabulary size are already aligned.")
        
        self.console.print()

        # --- Verification Step (Still useful) ---
        text_with_tokens = f"Here is my thought process: <think><cot_explication> I will analyze the situation. </think><think>{mapping_thouhts['<cot_ton>']} I will be neutral. </think> Now, I conclude that the answer is correct.</think><think>{mapping_thouhts['<cot_intention>']} I want to provide a clear explanation. </think><think>{mapping_thouhts['<cot_categorie_list>']} Category1, Category2 </think><think>{mapping_thouhts['<cot_categorie_justification>']} These categories are relevant. </think><think>{mapping_thouhts['<cot_labels_list>']} Label1, Label2 </think><think>{mapping_thouhts['<cot_labels_justification>']} These labels are justified. </think><think>{mapping_thouhts['<cot_toxicite_note>']} 0.5 </think><think>{mapping_thouhts['<cot_toxicite_justification>']} The content is neutral. </think>"
        tokenized_output = self.tokenizer.tokenize(text_with_tokens)
        self.console.print("[bold]Verification of a special token:[/bold]")
        self.console.print(f"Tokenized example: '{text_with_tokens}'")
        self.console.print(f"-> [cyan]{tokenized_output}[/cyan]")
        self.console.print()


        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >               Load From Local Checkpoint               >
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if self.local_checkpoint:
            self.console.rule("[bold yellow]Loading LoRA Adapters from Local Checkpoint")
            try:
                self.console.print(f"Attempting to load adapters from: [cyan]{self.local_checkpoint}[/cyan]")
                
                # The base model is already loaded, now we apply the LoRA adapters on top.
                # `is_trainable=False` is set for inference mode. If you want to continue training, set it to True.
                self.model = PeftModel.from_pretrained(self.model, self.local_checkpoint, is_trainable=False)
                
                # Optional: Merge the LoRA layers into the base model for faster inference.
                # This makes the model a standard transformer model again, but with the learned weights.
                # After merging, you can no longer train the LoRA layers.
                # self.console.print("Merging LoRA adapters into the base model...")
                # self.model = self.model.merge_and_unload()
                
                self.console.print(f"[bold green]Successfully loaded LoRA adapters from '{self.local_checkpoint}'![/bold green]")
                
                # Ensure the final model is on the correct device
                self.model = self.model.to(self.device)
                self.console.print(f"Model with adapters moved to device: [cyan]{self.device}[/cyan]")

            except Exception as e:
                self.console.print(f"[bold red]Error loading local checkpoint '{self.local_checkpoint}': {e}[/bold red]")
                self.console.print("[yellow]Warning: Continuing with the base model only.[/yellow]")
            self.console.print()

    # ╭────────────────────────────────────────────────────────╮
    # │                  Dataset Preparation                   │
    # ╰────────────────────────────────────────────────────────╯

    ### Prepares the dataset for training by loading it and applying the formatting function.
    def prepare_dataset(self):
        split = load_dataset(self.dataset_path,
                             name=self.dataset_name,
                             trust_remote_code=True)
        format_fn = partial(formatting_func,
                            text_field=self.text_field,
                            cot_field=self.cot_field,
                            label_field=self.label_field,
                            tokenizer=self.tokenizer,
                            remove_thoughts=self.remove_thoughts)
        self.dataset = {
            k: v.map(format_fn, load_from_cache_file=False)
                 .remove_columns([c for c in v.column_names
                                  if c not in ("prompt", "completion")])
            for k, v in split.items()
        }
        self.console.print(f"[green]Train:[/] {len(self.dataset['train'])}  "
                           f"[cyan]Test:[/] {len(self.dataset['test'])}")

    # ╭────────────────────────────────────────────────────────╮
    # │                   Training handling                    │
    # ╰────────────────────────────────────────────────────────╯

    ### Creates and returns a LoraConfig based on the provided parameters.
    def _get_lora_config(self, task_type=TaskType.CAUSAL_LM, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05, bias: str = "none", target_modules: list = None, **kwargs) -> LoraConfig:
        """Creates and returns a LoraConfig."""
        if target_modules is None:
            target_modules = self.default_lora_target_modules
        
        config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,               # Use `print(self.model)` to see the available modules and the new modules added by the model
            # modules_to_save=["lm_head", "embed_tokens"], # make sure to save the lm_head and embed_tokens as you train the special tokens (c.f. https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora?hl=fr)
            **kwargs
        )
        self.console.print(f"Using LoRA Config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, targets={target_modules}")
        return config

    ### Creates and returns TrainingArguments based on the provided parameters.
    def _get_training_args(self, output_dir: Path = Path("output"), **kwargs) -> SFTConfig:
        """Creates and returns SFTConfig."""
        default_args = {
            ### Core Optimization Parameters
            "learning_rate": 2e-4,              # Initial learning rate for the optimizer, common for LoRA training
            "num_train_epochs": 5,              # Number of epochs to train the model
            "warmup_steps": 20,                 # Number of warmup steps for the learning rate scheduler (We get inspiration from https://github.com/deepseek-ai/DeepSeek-Coder/tree/main/finetune)
            "weight_decay": 0.01,               # Weight decay for L2 regularization
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",      # Learning rate scheduler type
            "deepspeed": "utils/ds_config.json",# Path to DeepSpeed configuration file (if using DeepSpeed)
            "max_length": 1024,                 # Set a power of 2 that is higher than the maximum sequence length of your dataset (here 995)

            ### Precision & Device
            "fp16": True,                       # Enable mixed precision training (float16)
            "bf16": False,
            "torch_compile": False,             # It can improve memory efficiency and speed. But use only if your GPU/driver stack supports it without bugs.
            "gradient_checkpointing": True,     # This trades computation for memory by not storing intermediate activations.
            "gradient_checkpointing_kwargs": {"use_reentrant": False}, 

            ### Saving & Logging
            "logging_dir": f"runs/exp_{self.dataset_name}_{self.model_name.split('/')[-1]}-{'-'.join(self.remove_thoughts)}",        # Directory for TensorBoard logs (launch `tensorboard --logdir runs/exp_X` and open it in your browser at http://localhost:6006)
            "logging_strategy": "steps",        # How often to log training metrics
            "logging_steps": 50,                # Log every 50 steps
            "save_strategy": "steps",           # Save model every few steps
            "save_steps": 50,
            "save_total_limit": 3,              # Limit the number of saved checkpoints to save space
            "run_name": "QLoRA_Training",       # Name of the run for logging purposes
            "report_to": "tensorboard",         # Reporting backend, can be "wandb", "tensorboard", etc.

            ### Evaluation
            "eval_strategy": "steps",               # Evaluate model every few epoch
            "eval_steps": 50,                      # Evaluate every 50 steps
            "load_best_model_at_end": True,         # Load the best model at the end of training based on evaluation metrics
            "metric_for_best_model": "eval_loss",   # Metric to use for determining the best model
            "greater_is_better": False,             # For loss, lower is better

            ### Batch Size Control
            "per_device_train_batch_size": 3,   # Batch size per device during training
            "per_device_eval_batch_size": 2,    # Batch size per device during evaluation
            "auto_find_batch_size": False,      # Automatically find the best batch size
            "gradient_accumulation_steps": 32,  # Number of steps to accumulate gradients before updating model weights
                                                # gradient_accumulation_steps = ​Batch_target​​ / Batch_GPU -> acc = 32 for GPU_batch=2×2 GPUs ⇒ 128

            ### Miscellaneous
            "remove_unused_columns": False,     # Drop unused columns in the dataset
            "seed": 42,                         # Random seed for reproducibility
        }
        # Override defaults with provided kwargs
        final_args = {**default_args, **kwargs}

        # Set fp16/bf16 based on compute_dtype if not explicitly set
        if 'fp16' not in kwargs and self.bnb_config.bnb_4bit_compute_dtype == torch.float16:
            final_args['fp16'] = True
        if 'bf16' not in kwargs and self.bnb_config.bnb_4bit_compute_dtype == torch.bfloat16:
            final_args['bf16'] = True
        
        self.console.print(f"Using SFTConfig: {final_args}")
        return SFTConfig(output_dir=output_dir, **final_args)

    ### Trains the model using SFTTrainer with the provided dataset and configurations.
    def train(self, peft_config_kwargs: Optional[Dict[str, Any]] = None, training_args_kwargs: Optional[Dict[str, Any]] = None):
        """
        Fine-tunes the model using SFTTrainer.

        Args:
            peft_config_kwargs (dict, optional): Keyword arguments for LoraConfig.
            training_args_kwargs (dict, optional): Keyword arguments for TrainingArguments.
        """
        if not self.model or not self.tokenizer:
            self.console.print("[bold red]Model or tokenizer not loaded. Cannot train.[/bold red]")
            return

        # Generate an example before training using `generate_text`` (take the first example from the eval dataset)
        if os.environ.get("RANK", "0") == "0":
            example = self.dataset["test"][0]
            self.console.print('\n')
            self.console.rule("[bold yellow]Example before training")
            self.console.print('\n')
            self.generate_text(prompt=example["prompt"], max_new_tokens=1024)
            self.console.print('\n')

        _peft_config_kwargs = peft_config_kwargs if peft_config_kwargs is not None else {}
        peft_config = self._get_lora_config(**_peft_config_kwargs)

        _training_args_kwargs = training_args_kwargs if training_args_kwargs is not None else {}
        training_arguments = self._get_training_args(**_training_args_kwargs)

        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)
        
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            args= training_arguments,
            peft_config=peft_config,
            processing_class=self.tokenizer,   # NEW NAME (≥ TRL 0.16)
            callbacks=[early_stopping_callback],
        )

        self.console.rule("[bold cyan]Starting Training", style="cyan")
        train_result = self.trainer.train()
        
        self.console.print(f"[bold green]Training completed![/bold green]")
        self.console.print(f"Training metrics: {train_result.metrics}")

        # Save the trained LoRA adapters
        adapter_save_path = Path(training_arguments.output_dir) / "final_adapters"
        self.trainer.save_model(adapter_save_path) # Saves LoRA adapters
        self.console.print(f"LoRA adapters saved to: {adapter_save_path}")

        # Generate an example after training using `generate_text`` (take the first example from the eval dataset)
        if os.environ.get("RANK", "0") == "0":
            self.console.print('\n')
            self.console.rule("[bold yellow]Example after training")
            self.console.print('\n')
            self.generate_text(prompt=example["prompt"], max_new_tokens=1024)
            self.console.print('\n')


    # ╭────────────────────────────────────────────────────────╮
    # │                       Inference                        │
    # ╰────────────────────────────────────────────────────────╯

    ### Generates text using the loaded model based on a given prompt.
    def generate_text(self, prompt: str, max_new_tokens: int = 1024, do_sample: bool = True, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.1, **kwargs):
        """
        Generates text using the loaded model.

        Args:
            prompt (str): The input prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.
            do_sample (bool): Whether to use sampling.
            temperature (float): Softmax temperature for sampling.
            top_p (float): Nucleus sampling probability.
            top_k (int): Top-k sampling.
            repetition_penalty (float): Repetition penalty.
            **kwargs: Additional arguments for model.generate().

        Returns:
            str: The generated text.
        """
        if not self.model or not self.tokenizer:
            self.console.print("[bold red]Model or tokenizer not loaded. Call `load_tokenizer_and_model()` first.[/bold red]")
            return None

        self.console.print('\n')
        self.console.rule("[bold yellow]Prompt")
        self.console.print('\n')
        self.console.print(Panel.fit(prompt, title="Prompt", style="italic white on black"))
        self.console.print('\n')

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        default_generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        # Override defaults with any kwargs passed to the function
        generation_params = {**default_generation_kwargs, **kwargs}

        outputs = self.model.generate(**inputs, **generation_params)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        self.console.rule("[bold magenta]Model Output")
        self.console.print('\n')
        self.console.print(Panel.fit(generated_text, title="Generated Text", style="bold white on black"))
        return generated_text
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Finetuning Utility")

    # Dataset and model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="Base model name")
    parser.add_argument("--dataset_path", type=str, default="Naela00/ToxiFrenchFinetuning", help="HuggingFace dataset repo path")
    parser.add_argument("--dataset_name", type=str, required=False, help="Dataset config name (e.g. 'odal')")
    parser.add_argument("--local_checkpoint", type=str, default=None, help="Path to a local checkpoint to load the model from")

    # Dataset fields
    parser.add_argument("--text_field", type=str, default="content", help="Field name containing the input text")
    parser.add_argument("--cot_field", type=str, default="cot_text", help="Field name containing the chain-of-thought")
    parser.add_argument("--label_field", type=str, default="literal_conclusion_annotator", help="Field name for the label")
    parser.add_argument("--remove_thoughts", type=str, default="<cot_intention>,<cot_categorie_list>,<cot_labels_list>", help="Comma-separated list of thought tokens to remove from the generated text")

    # Mode
    parser.add_argument("--mode", type=str, choices=["train", "generate"], default="train", help="Operation mode")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text for generation")

    # Training options
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=5, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=5, help="Evaluation batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation step frequency")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging step frequency")
    parser.add_argument("--deepspeed_config", type=str, default="utils/ds_config.json", help="Path to DeepSpeed config file")

    # LoRA options
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--lora_bias", type=str, default="none", help="LoRA bias type")
    
    # Generation options
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")

    # Proxy and quantization
    parser.add_argument("--use_proxy", type=bool, default=True, help="Use proxy for model downloading")

    args = parser.parse_args()


    handler = QLoRAModelHandler(
        model_name=args.model,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        text_field=args.text_field,
        cot_field=args.cot_field,
        label_field=args.label_field,
        remove_thoughts=args.remove_thoughts.split(","),
        use_proxy=args.use_proxy,
        local_checkpoint=args.local_checkpoint,
    )

    handler.prepare_dataset()

    if args.mode == "train":
        handler.train(
            peft_config_kwargs={
                "r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "bias": args.lora_bias,
            },
            training_args_kwargs={
                "num_train_epochs": args.epochs,
                "per_device_train_batch_size": args.batch_size,
                "per_device_eval_batch_size": args.eval_batch_size,
                "learning_rate": args.lr,
                "output_dir": args.output_dir if args.output_dir else f"output_{args.dataset_name}_{args.model.split('/')[-1]}_{'-'.join(handler.remove_thoughts)}",
                "eval_steps": args.eval_steps,
                "logging_steps": args.logging_steps,
                "deepspeed": args.deepspeed_config,
            }
        )

    elif args.mode == "generate":
        if args.prompt:
            handler.generate_text(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty
            )
        else:
            print("You must specify a --prompt in generate mode.")

