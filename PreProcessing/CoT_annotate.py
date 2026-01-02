# ╔════════════════════════════════════════════════════════╗
# ║                   GPT CoT annotation                   ║
# ╚════════════════════════════════════════════════════════╝

# +--------------------------------------------------------+
# |  This script launches batch annotation jobs using the  |
# |                    OpenAI GPT API.                     |
# +--------------------------------------------------------+

import argparse
from pathlib import Path
import pandas as pd
import openai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich import box

from utils.openai_api_handling import OpenAIResponseHandler
from utils.analysis_headers import PROMPTS_HEADERS

# Initialize Rich console
console = Console()

def log_header():
    console.rule("[bold cyan]Batch Annotation Launcher", style="cyan")
    console.print("This script will annotate the prioritized dataset using GPT batch API.", style="italic")

def show_paths_table(CSV_IN, CHECKPOINT_CSV, BATCH_PATH, OUTPUT_PATH):
    table = Table(title="Paths Overview", box=box.SIMPLE, show_edge=True)
    table.add_column("File", style="bold green")
    table.add_column("Path", style="dim")

    table.add_row("Input CSV", str(CSV_IN))
    table.add_row("Checkpoint CSV", str(CHECKPOINT_CSV))
    table.add_row("Batch File", str(BATCH_PATH))
    table.add_row("Output File", str(OUTPUT_PATH))

    console.print(table)

def main(args):

    if args.step not in PROMPTS_HEADERS:
        console.print(f"[bold red] Invalid step '{args.step}'[/bold red]")
        console.print(f"Available steps: {list(PROMPTS_HEADERS.keys())}")
        raise ValueError(f"Invalid step '{args.step}'.")

    # =========================================================================
    # Paths & Config
    # =========================================================================
    ROOT_DIR = Path("..")
    DATA_DIR = ROOT_DIR / "data"
    model_name = args.model
    step_name = args.step

    # Direct path to the prioritized dataset
    CSV_IN = DATA_DIR / "ForumData" / "anonymous_forum_prioritized.csv"

    CHECKPOINT_CSV = DATA_DIR / "ForumData" / "anonymous_forum_prioritized_annotated.csv"

    
    # Batch files now identify by step and model
    BATCH_PATH     = DATA_DIR / "ForumData" / f"batch_input_{step_name}_{model_name}.jsonl"
    OUTPUT_PATH    = DATA_DIR / "ForumData" / f"batch_results_{step_name}_{model_name}.jsonl"
    
    API_KEY_PATH   = DATA_DIR / "confidential" / "GPT_API.txt"

    log_header()
    show_paths_table(CSV_IN, CHECKPOINT_CSV, BATCH_PATH, OUTPUT_PATH)

    # =========================================================================
    # Initialize handler
    # =========================================================================
    handler = OpenAIResponseHandler(
        CSV_IN=CSV_IN,
        CHECKPOINT_CSV=CHECKPOINT_CSV,
        STEP=step_name,
        USE_PROXY=args.proxy,
        MODEL=model_name,
        MAX_CONTENTS=args.max_contents,
        batch_path=BATCH_PATH,
        max_tokens=args.max_tokens,
        path_api_key=API_KEY_PATH,
        output_path=OUTPUT_PATH,
        exclude_batchids=args.exclude_batchids,
    )

    # =========================================================================
    # Run Annotation Pipeline
    # =========================================================================
    handler.load_data_and_resume()

    if not args.info:
        console.print("[bold green] Starting annotation process...[/bold green]")
        
        # 
        
        if not args.from_batchid: 
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                progress.add_task(description="Creating JSONL batch...", total=None)
                handler.create_json_batch()

                progress.add_task(description="Uploading batch to OpenAI...", total=None)
                handler.upload_batch()

                progress.add_task(description="Submitting batch job...", total=None)
                handler.submit_batch()
        else:
            handler.batch = openai.batches.retrieve(args.from_batchid)
            console.print(f"[cyan] Resuming from existing batch ID: [bold]{args.from_batchid}[/bold][/cyan]")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            progress.add_task(description="Waiting for and parsing results...", total=None)
            results = handler.download_and_parse_results()

        handler.merge_and_save(results)
        handler.summarize_tokens(results)

        console.print(Panel.fit(f"[bold green] Annotation complete.[/bold green]\n\n", border_style="green"))

# =========================================================================
# Entry Point
# =========================================================================
if __name__ == "__main__":
    console.print("[bold]Example of usage:[/bold]")
    console.print("python launch_batches_GPT_api.py --step toxicite_score --max_contents 1000", style="dim")
    print()

    parser = argparse.ArgumentParser(description="Annotate the prioritized dataset using OpenAI GPT API batch mode.")

    parser.add_argument("--step", type=str, required=True, help=f"Annotation step (must match a key in PROMPTS_HEADERS). Available steps: {list(PROMPTS_HEADERS.keys())}")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--proxy", type=str, default="socks5h://127.0.0.1:1080", help="Proxy URL (if needed)")
    parser.add_argument("--max_contents", type=int, default=None, help="Maximum number of rows to annotate")
    parser.add_argument("--max_tokens", type=int, default=250, help="Max tokens for completion")
    parser.add_argument("--from_batchid", type=str, default="", help="Batch ID to resume from")
    parser.add_argument("--info", type=bool, default=False, help="Print info about the batch before running")
    parser.add_argument("--exclude_batchids",nargs="*",default=[],help="List of OpenAI batch IDs whose input msg_ids should be excluded")

    args = parser.parse_args()
    main(args)