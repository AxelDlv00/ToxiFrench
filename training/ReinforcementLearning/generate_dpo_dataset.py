# ╔════════════════════════════════════════════════════════╗
# ║ CLASS TO GENERATE AUTOMATICALLY THE DPO DATASET FROM A ║
# ║                   CHECKPOINTED MODEL                   ║
# ╚════════════════════════════════════════════════════════╝

import os
import sys
import torch
import argparse
import json
import glob
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
import numpy as np
from datasets import load_dataset, concatenate_datasets

import os
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[2] 
finetuning_path = project_root / "training" / "FineTuning"

sys.path.append(str(project_root))
sys.path.append(str(finetuning_path))

from training.FineTuning.finetuning_handler import QLoRAModelHandler
from training.FineTuning.utils.dataset_handling import DatasetPreparer

# ╔════════════════════════════════════════════════════════╗
# ║                    DatasetPreparer                     ║
# ╚════════════════════════════════════════════════════════╝

class DPODatasetPreparer(DatasetPreparer):
    def __init__(self, *args, undersampling=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.undersampling = undersampling

    def prepare(self, dataset_path, split_name="train", cache=False, remove_columns=["text", "prompt", "completion"]):
        ds = load_dataset(dataset_path, split=split_name)

        if self.undersampling:
            labels = ds[self.label_field]
            unique_labels, counts = np.unique(labels, return_counts=True)
            min_count = min(counts)
            
            datasets_to_combine = []
            for label in unique_labels:
                label_ds = ds.filter(lambda x: x[self.label_field] == label)
                
                shuffled_label_ds = label_ds.shuffle(seed=self.seed)
                downsampled_ds = shuffled_label_ds.select(range(min_count))
                
                datasets_to_combine.append(downsampled_ds)
            
            ds = concatenate_datasets(datasets_to_combine)
            ds = ds.shuffle(seed=self.seed)
            
            print(f"[INFO] Undersampling appliqué : {len(ds)} exemples conservés ({min_count} par classe).")

        self.oversampling = False
        
        ds = ds.map(
            self.formatting_func,
            remove_columns=ds.column_names, 
            desc="Formatting DPO dataset",
            load_from_cache_file=cache
        )

        tokenized_ds = ds.map(
            self.tokenize_function,
            batched=False,
            remove_columns=remove_columns,
            desc="Tokenizing DPO dataset",
            load_from_cache_file=cache
        )

        return tokenized_ds

# ╔════════════════════════════════════════════════════════╗
# ║                     DPOGenerator                       ║
# ╚════════════════════════════════════════════════════════╝

class DPOGenerator(QLoRAModelHandler):
    def __init__(self, **kwargs):
        # Force inference mode
        kwargs["mode"] = "inference"
        super().__init__(**kwargs)
        
        # Critical for generation: Padding must be on the LEFT
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_dpo_dataset(self, 
                             n_generations=2, 
                             batch_size=4, 
                             output_path="dpo_data.jsonl", 
                             **gen_kwargs):
        
        self.model.eval()
        self.console.rule("[bold green]DPO Batch Data Generation[/bold green]")
        self.console.print(f"Generations per prompt: {n_generations}")
        self.console.print(f"Batch Size: {batch_size}")

        # Prepare Dataset
        preparer = DPODatasetPreparer(
            self.tokenizer, 
            self.text_field, 
            self.cot_fields, 
            self.label_field,
            undersampling=True,
            seed=42
        )
        dataset = preparer.prepare(
            self.dataset_path, 
            split_name="train", 
            remove_columns=[] 
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            
            # Use tqdm only on main process to avoid clutter
            iterator = tqdm(range(0, len(dataset), batch_size), 
                          desc=f"Rank 0", 
                          disable=False)

            for i in iterator:
                batch_samples = dataset[i : i + batch_size]
                base_prompts = batch_samples["prompt"]
                prompts_for_model = [p + "<think>\n" for p in base_prompts]
                references = batch_samples["completion"]
                inputs = self.tokenizer(
                    prompts_for_model, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(self.device)

                input_len = inputs.input_ids.shape[1]

                gen_config = {
                    "max_new_tokens": gen_kwargs.get("max_new_tokens", 1024),
                    "temperature": gen_kwargs.get("temperature", 0.7),
                    "top_p": gen_kwargs.get("top_p", 0.9),
                    "do_sample": gen_kwargs.get("do_sample", True),
                    "num_return_sequences": n_generations, 
                    "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.1),
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_config)

                generated_tokens = outputs[:, input_len:]
                
                decoded_candidates = self.tokenizer.batch_decode(
                    generated_tokens, 
                    skip_special_tokens=False
                )

                
                for j, (prompt, ref) in enumerate(zip(base_prompts, references)):
                    start_idx = j * n_generations
                    end_idx = start_idx + n_generations
                    
                    candidates = decoded_candidates[start_idx:end_idx]
                    
                    record = {
                        "prompt": prompt,
                        "reference": ref,
                        "candidates": candidates,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ╔════════════════════════════════════════════════════════╗
# ║                   Main Execution                       ║
# ╚════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # Launch with : python generate_dpo_dataset.py --config config_dpo_gen.json
    parser = argparse.ArgumentParser(description="DPO Dataset Generation")
    parser.add_argument("--config", type=str, default="config_dpo_gen.json", help="Path to the config file")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = json.load(f)

    dpo_cfg = config.get("dpo_params", {})
    n_gen = dpo_cfg.get("n_gen", 2)
    batch_size = args.batch_size or dpo_cfg.get("batch_size", 4)
    output_path = dpo_cfg.get("output_path", "dpo_candidates.jsonl")

    model_params = config.get("model_params", {})
    if args.checkpoint:
        model_params["local_checkpoint"] = args.checkpoint
    
    generator = DPOGenerator(
        **model_params,
        **config.get("dataset_params", {}),
        bnb_config_kwargs=config.get("bnb_params", {})
    )
    gen_params = config.get("training_params", {})
    
    generator.generate_dpo_dataset(
        n_generations=n_gen,
        batch_size=batch_size,
        output_path=output_path,
        **gen_params
    )