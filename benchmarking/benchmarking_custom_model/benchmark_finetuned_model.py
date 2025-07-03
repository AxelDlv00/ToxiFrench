"""
╔════════════════════════════════════════════════════════╗
║                 Benchmarking our model                 ║
╚════════════════════════════════════════════════════════╝

This script benchmarks our finetuned models on ToxiFrench dataset.
e.g. torchrun --nproc_per_node=2 benchmark_finetuned_model.py --dataset_name oeal
e.g. torchrun --nproc_per_node=2 benchmark_finetuned_model.py --dataset_name odal --tuned dpo_ --checkpoint final_dpo_adapters --remove_thoughts ""
"""

# ╔════════════════════════════════════════════════════════╗
# ║                       Libraries                        ║
# ╚════════════════════════════════════════════════════════╝

import pandas as pd
import torch
from pathlib import Path
import sys
from tqdm import tqdm
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import argparse
import os

ROOT = Path('../..')
sys.path.append(str(ROOT))
from reinforcement_learning.DPO_handler import DPOHandler

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ╭────────────────────────────────────────────────────────╮
# │         INITIALIZE THE DISTRIBUTED ENVIRONMENT         │
# ╰────────────────────────────────────────────────────────╯

# torchrun will automatically set these environment variables
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Running DDP on rank {rank} of {world_size}.")
else:
    # This allows the script to run in a non-distributed way (e.g., on a single GPU)
    rank = 0
    world_size = 1
    print("Not running in a distributed environment.")

# Helper to prevent multiple processes from printing tqdm bars
is_main_process = (rank == 0)

# ╔════════════════════════════════════════════════════════╗
# ║                    Global Variables                    ║
# ╚════════════════════════════════════════════════════════╝

parser = argparse.ArgumentParser(description="Benchmarking Custom Model Utility")
parser.add_argument("--dataset_name",  type=str, default="real", help="Name of the dataset to use")
parser.add_argument("--tuned", type=str, default="", help="Tuned model identifier")
parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-4B", help="Base model name")
parser.add_argument("--remove_thoughts", type=str, default='<cot_intention>,<cot_categorie_list>', help="Comma-separated list of thoughts to remove from the prompt")
parser.add_argument("--checkpoint", type=str, default="final_adapters", help="Checkpoint name for the LoRA adapter")
parser.add_argument("--batch_size", type=int, default=80, help="Batch size for DataLoader (per device)")

args = parser.parse_args()

dataset_name = args.dataset_name
tuned = args.tuned
model_name = args.base_model_name
remove_thoughts = args.remove_thoughts.split(',')
checkpoint = args.checkpoint
batch_size = args.batch_size
split = "test"

DATA_DIR = ROOT / "data"

console = Console()

step = 'finetuning'

file_path = ROOT / step / f'{tuned}output_{dataset_name}_{model_name.split("/")[-1]}{"_" if remove_thoughts and remove_thoughts[0] else ""}{"-".join(remove_thoughts)}' / checkpoint
output_path =DATA_DIR / "benchmark" /  "benchmark_our_custom_model" / f'{tuned}output_{dataset_name}_{model_name.split("/")[-1]}{"_" if remove_thoughts else ""}{"-".join(remove_thoughts)}_{rank}.csv'
if not file_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

# ╔════════════════════════════════════════════════════════╗
# ║                        Handler                         ║
# ╚════════════════════════════════════════════════════════╝

handler = DPOHandler(
    model_name=model_name,
    mode="generate",
    dataset_name=dataset_name,
    local_checkpoint=str(file_path),
    use_proxy=True,
)

# ╔════════════════════════════════════════════════════════╗
# ║                        Dataset                         ║
# ╚════════════════════════════════════════════════════════╝

handler.prepare_dataset(dpo=False)

# ╔════════════════════════════════════════════════════════╗
# ║                      Sample data                       ║
# ╚════════════════════════════════════════════════════════╝

def custom_collate_fn(batch):
    """
    A custom collate function that filters out samples that are None.
    This is useful when your dataset might have corrupted or missing data.
    """
    # Filter out None values from the batch
    original_size = len(batch)
    batch = [item for item in batch if item is not None]
    
    # If the entire batch was problematic, you might return None or an empty dict
    if not batch:
        if original_size > 0:
            print(f"Warning: A batch of size {original_size} was filtered down to 0. All samples were None.")
        return None # Or however your loop handles this
        
    # Now, use the default collate function on the cleaned batch
    return torch.utils.data.default_collate(batch)

print(f"Selecting the '{split}' split from the dataset for benchmarking.")
benchmark_split_dataset = handler.dataset[split]
print(f"'{split}' split contains {len(benchmark_split_dataset)} examples.")

sampler = DistributedSampler(benchmark_split_dataset, num_replicas=world_size, rank=rank, shuffle=False)
loader = DataLoader(
    benchmark_split_dataset, 
    batch_size=batch_size, 
    sampler=sampler, 
    collate_fn=custom_collate_fn
)

# ╔════════════════════════════════════════════════════════╗
# ║                    Generate_answers                    ║
# ╚════════════════════════════════════════════════════════╝

all_results = []
for batch_data in tqdm(loader, desc=f"Rank {rank} processing '{split}' split", disable=not is_main_process):
    if batch_data is None:
        continue
    prompts_batch = batch_data['prompt']
    ids_batch = batch_data['msg_id']
    label = batch_data['label']
    
    generated_texts = handler.generate_text_batch(prompts_batch)
    
    for i in range(len(generated_texts)):
        all_results.append({
            'id': ids_batch[i],
            'label': label[i],
            'prompt': prompts_batch[i],
            'generated_text': generated_texts[i]
        })

# ╔════════════════════════════════════════════════════════╗
# ║                    Post-processing                     ║
# ╚════════════════════════════════════════════════════════╝

df = pd.DataFrame(all_results)
df['prediction'] = df['generated_text'].apply(lambda x: x.split('</think>')[-1].replace('En conclusion, ce message est-il toxique ?', '').strip().split('<|im_end|>')[0].strip())
df['prediction'] = df['prediction'].apply(lambda x: 1 if 'oui' in x.lower() else 0)

console.print(f"[bold green]Total samples processed: {len(df)}[/bold green]")
console.print(f"[bold blue]The predictions are distributed as follows:[/bold blue]")
console.print(f"[bold yellow]Label 0 (Non-toxique):[/bold yellow] {df['prediction'].value_counts().get(0, 0)}")
console.print(f"[bold red]Label 1 (Toxique):[/bold red] {df['prediction'].value_counts().get(1, 0)}")

df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Results saved to: {output_path}")
