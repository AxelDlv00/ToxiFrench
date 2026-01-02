# ╔════════════════════════════════════════════════════════╗
# ║        DPO launcher for Reinforcement Learning         ║
# ╚════════════════════════════════════════════════════════╝

import os
import re
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from functools import partial
import argparse
import json
import sys

# Rich for beautiful logging
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# HF & TRL Libraries
from transformers import TrainerCallback
from peft import LoraConfig, TaskType, PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, concatenate_datasets
import torch.distributed as dist
from accelerate import PartialState

# Setup Paths
current_file = Path(__file__).resolve()
project_root = current_file.parents[2] 
finetuning_path = project_root / "training" / "FineTuning"

sys.path.append(str(project_root))
sys.path.append(str(finetuning_path))

# Custom handlers
from training.FineTuning.finetuning_handler import QLoRAModelHandler
from training.FineTuning.utils.dataset_handling import DatasetPreparer

# ╔════════════════════════════════════════════════════════╗
# ║                   DPOModelHandler                      ║
# ╚════════════════════════════════════════════════════════╝

class DPOModelHandler(QLoRAModelHandler):
    def __init__(self, **kwargs):
        kwargs["mode"] = kwargs.get("mode", "train")
        super().__init__(**kwargs)
        
        self.chosen_field = kwargs.get("chosen_field", "chosen")
        self.rejected_field = kwargs.get("rejected_field", "rejected")
        self.oversampling = kwargs.get("oversampling", False)
        
        if self.local_checkpoint and not isinstance(self.model, PeftModel):
            if PartialState().is_main_process:
                self.console.print(f"[bold yellow]Loading SFT Adapters for DPO from: {self.local_checkpoint}[/bold yellow]")
            
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.local_checkpoint, 
                is_trainable=True 
            )
            
            if PartialState().is_main_process:
                self.model.print_trainable_parameters()

    def train_dpo(self, peft_kwargs: dict = None, training_kwargs: dict = None, custom_kwargs: dict = None):
        if PartialState().is_main_process:
            self.console.rule("[bold magenta]Preparing DPO Training[/bold magenta]")

        ds = load_dataset(
            custom_kwargs.get("dataset_path", "AxelDlv00/ToxiFrench"), 
            custom_kwargs.get("dataset_config", 'dpo'),
            split=custom_kwargs.get("split", "train")
        )

        dpo_config = self._build_dpo_config(training_kwargs)

        if isinstance(self.model, PeftModel):
            if PartialState().is_main_process:
                self.console.print("[bold cyan]Detected loaded PeftModel. Continuing training on existing adapters.[/bold cyan]")
            peft_config = None
        else:
            peft_config = self._build_lora_config(peft_kwargs) if peft_kwargs else None

        self.model.enable_input_require_grads()
        if dpo_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=None, 
            args=dpo_config,
            train_dataset=ds,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )

        if PartialState().is_main_process:
            self.console.print("[bold cyan]Starting DPO Training Loop...[/bold cyan]")
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if trainable_params == 0:
                self.console.print("[bold red]WARNING: No trainable parameters found! check is_trainable=True in loading.[/bold red]")
            else:
                self.console.print(f"[bold yellow]Trainable parameters: {trainable_params}[/bold yellow]")
        
        self.trainer.train(resume_from_checkpoint=None)
        
        final_path = Path(dpo_config.output_dir) / "final_dpo_adapters"
        self.trainer.save_model(final_path)
        
        if PartialState().is_main_process:
            self.console.print(f"[bold green]Model saved at: {final_path}[/bold green]")

    def _build_dpo_config(self, kwargs) -> DPOConfig:
        default_args = {
            "output_dir": "./output_dpo",
            "beta": 0.1,  
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": True,
            "logging_steps": 1,
            "max_prompt_length": 256, 
            "max_length": 1024,       
            "remove_unused_columns": False,
            "bf16": torch.cuda.is_bf16_supported(),
        }
        
        if kwargs:
            default_args.update(kwargs)
            
        return DPOConfig(**default_args)

# ╔════════════════════════════════════════════════════════╗
# ║                      Entry Point                       ║
# ╚════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # Example Launch: 
    # accelerate launch launch_dpo.py --config config/rec_soap_dpo_augmented.json

    parser = argparse.ArgumentParser(description="DPO Training Launcher")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config_dpo.json", 
        help="Path to the config file"
    )
    args = parser.parse_args()
    
    if PartialState().is_main_process:
        console = Console()
    else:
        class DummyConsole:
            def print(self, *args, **kwargs): pass
            def rule(self, *args, **kwargs): pass
        console = DummyConsole()

    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[bold red]Error:[/bold red] The file {args.config} was not found.")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    bnb_kwargs = config.get("bnb_params", {})
    if bnb_kwargs.get("bnb_4bit_compute_dtype") == "float16":
        bnb_kwargs["bnb_4bit_compute_dtype"] = torch.float16
    elif bnb_kwargs.get("bnb_4bit_compute_dtype") == "bfloat16":
        bnb_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16

    handler = DPOModelHandler(
        **config.get("model_params", {}),
        **config.get("dataset_params", {}),
        bnb_config_kwargs=bnb_kwargs
    )

    mode = config.get("model_params", {}).get("mode", "train")
    if mode == "train":
        handler.train_dpo(
            peft_kwargs=config.get("lora_params"),
            training_kwargs=config.get("training_params"),
            custom_kwargs=config.get("training_params_custom")
        )
    else:
        console.print("[yellow]Warning:[/yellow] Currently, only 'train' mode is implemented for DPO.")