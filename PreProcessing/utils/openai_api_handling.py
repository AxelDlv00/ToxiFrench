# ==========================================================
# OpenAI API Handling Utilities
# ==========================================================

from __future__ import annotations
from typing import List, Dict, Any
import os
import json
import time
from pathlib import Path
import numpy as np

import pandas as pd
import openai

from utils.analysis_headers import PromptType, PROMPTS_HEADERS

from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
# ----------------------------------------------------------
# Pricing constants (USD per 1K tokens)
# ----------------------------------------------------------
PRICE_PER_K_PROMPT = 0.0005
PRICE_PER_K_COMPLETION = 0.0015

# ----------------------------------------------------------
# Batch File Creation for Pipeline Steps
# ----------------------------------------------------------

def create_batch_file(
    MODEL: str,
    prompt: PromptType,
    fields: Dict[str, str],
    max_tokens: int = 150
) -> Dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.create_prompt(fields)}
        ],
        "max_tokens": max_tokens
    }

# ==========================================================
# Class for Handling OpenAI API Responses
# ==========================================================

class OpenAIResponseHandler:

    def __init__(
        self,
        CSV_IN: Path,
        CHECKPOINT_CSV: Path,
        STEP: str,
        USE_PROXY: str = "",
        MODEL: str = "gpt-4o-mini",
        MAX_CONTENTS: int | None = None,
        batch_path: Path = Path("batch.jsonl"),
        max_tokens: int = 150,
        path_api_key: Path = Path("path/to/api_key.txt"),
        output_path: Path = Path("output.jsonl"),
        exclude_batchids: list[str] = [],
    ):
        self.CSV_IN = CSV_IN
        self.CHECKPOINT_CSV = CHECKPOINT_CSV
        self.STEP = STEP
        self.MAX_CONTENTS = MAX_CONTENTS
        self.batch_path = batch_path
        self.max_tokens = max_tokens
        self.path_api_key = path_api_key
        self.output_path = output_path
        self.MODEL = MODEL
        self.batch = None
        self.uploaded_file_id = None
        self.console = Console()
        self.exclude_batchids = exclude_batchids

        # Load full dataset
        self.df_all = pd.read_csv(CSV_IN, dtype={"msg_id": str})
        self.df_before: pd.DataFrame | None = None
        self.df_checkpoint: pd.DataFrame | None = None

        log = ""
        if USE_PROXY:
            log += self.set_proxy(USE_PROXY)
        log += self.init_client()
        self.console.print(Panel(log, title="[bold green]Init", subtitle=""))

    def set_proxy(self, proxy_url: str):
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        log = f"Proxy set to: {proxy_url}\n"
        return log

    def init_client(self):
        openai.api_key = self.path_api_key.read_text().strip()
        log = "OpenAI API key set."
        return log

    def info_progress(self) -> None:
        table = Table(title="[bold cyan]Annotation Progress Summary", show_lines=True)

        table.add_column("Column", style="bold white")
        table.add_column("Prompt Tokens", justify="right")
        table.add_column("Unique Values", justify="right")
        table.add_column("To Annotate", justify="right")

        for column, value in PROMPTS_HEADERS.items():
            if column in self.df_before.columns:
                prompt_tokens = len(value.system_prompt.split())
                unique_vals = self.df_before[column].nunique()
                to_annotate = self.df_before[column].isna().sum() + (self.df_before[column] == "").sum()
                table.add_row(column, str(prompt_tokens), str(unique_vals), str(to_annotate))
            else:
                table.add_row(f"[dim]{column}[/dim]", "-", "-", "[red]Not annotated[/red]")

        self.console.print(table)

    def info_tokens(self) -> None:
        # Identify previous annotation steps
        index_step = list(PROMPTS_HEADERS.keys()).index(self.STEP)
        prev_steps_possible_columns = list(PROMPTS_HEADERS.keys())[:index_step]
        prev_steps = [col for col in self.df_checkpoint.columns
                    if col not in self.df_all.columns and col in prev_steps_possible_columns]

        # Start with system prompt tokens for the current step
        total_input_tokens = len(PROMPTS_HEADERS[self.STEP].system_prompt.split()) * len(self.df_checkpoint)

        # Create summary table
        table = Table(title=f"[bold cyan]Token Estimation for Step: [green]{self.STEP}[/green]", show_lines=True)
        table.add_column("Step", style="bold")
        table.add_column("System Prompt Tokens", justify="right")
        table.add_column("User Tokens", justify="right")

        for column in prev_steps:
            prompt_size = len(PROMPTS_HEADERS[column].system_prompt.split())
            user_tokens = self.df_checkpoint[column].dropna().astype(str).str.split().map(len).sum()
            total_input_tokens += prompt_size * len(self.df_checkpoint)
            total_input_tokens += user_tokens
            table.add_row(column, f'{prompt_size} (x {len(self.df_checkpoint)})', str(user_tokens))
        
        table.add_row(self.STEP, f'{len(PROMPTS_HEADERS[self.STEP].system_prompt.split())} (x {len(self.df_checkpoint)})', "-")

        # Display table and summary
        self.console.print(table)

        summary_panel = Panel.fit(
            f"[bold yellow]Current step:[/bold yellow] {self.STEP}\n"
            f"[bold yellow]Rows to annotate:[/bold yellow] {len(self.df_checkpoint)}\n"
            f"[bold green]Estimated total input tokens:[/bold green] ~{total_input_tokens}",
            title="[bold magenta]Total Estimate", border_style="magenta"
        )
        self.console.print(summary_panel)

    def get_msg_ids_from_batchids(self) -> set[str]:
        if not self.exclude_batchids:
            return set()

        msg_ids = set()
        for batch_id in self.exclude_batchids:
            try:
                batch = openai.batches.retrieve(batch_id)
                file_id = batch.input_file_id
                content = openai.files.content(file_id).read().decode("utf-8")
                for line in content.splitlines():
                    obj = json.loads(line)
                    custom_id = obj.get("custom_id", "")
                    if "_" in custom_id:
                        _, msg_id = custom_id.split("_", 1)
                        msg_ids.add(msg_id.strip())
            except Exception as e:
                self.console.print(f"[red]⚠ Failed to retrieve input for batch {batch_id}: {e}[/red]")
        return msg_ids

    def load_data_and_resume(self) -> None:
        """
        Load data from CSV_IN and filter out already processed rows based on CHECKPOINT_CSV and STEP.
        """
        
        if self.CHECKPOINT_CSV.exists():
            df = pd.read_csv(self.CHECKPOINT_CSV, dtype={"msg_id": str})
            self.df_before = pd.read_csv(self.CHECKPOINT_CSV, dtype={"msg_id": str})
            if self.STEP in df.columns:
                done_ids = df[
                    df[self.STEP].notna() & (df[self.STEP].astype(str).str.strip() != "")
                ]["msg_id"].unique()
                df = df[~df["msg_id"].isin(done_ids)]
            else:
                print(f"STEP '{self.STEP}' not found in checkpoint. Proceeding without filtering.")
        else:
            df = self.df_all.copy()
            df[self.STEP] = None
            self.df_before = df.copy()
            df.to_csv(self.CHECKPOINT_CSV, index=False)
        
        # Apply batch ID-based exclusions
        if self.exclude_batchids:
            excluded = self.get_msg_ids_from_batchids()
            n_before = len(df)
            df = df[~df["msg_id"].isin(excluded)]
            self.console.print(f"[cyan]✂ Excluded {len(excluded)} msg_ids from batch IDs ({n_before - len(df)} dropped)[/cyan]")

        if self.MAX_CONTENTS:
            df = df.sample(n=min(self.MAX_CONTENTS, len(df)), random_state=42)

        self.info_progress()
        self.df_checkpoint = df
        self.info_tokens()

    def create_json_batch(self) -> None:
        """
        Build a batch .jsonl file for the current STEP using rows in df_checkpoint.
        """
        assert self.df_checkpoint is not None, "Call load_data_and_resume() first."
        json_batch: dict[str, Any] = {}

        # Identify any previous annotation steps
        index_step = list(PROMPTS_HEADERS.keys()).index(self.STEP)
        prev_steps_possible_columns = list(PROMPTS_HEADERS.keys())[:index_step]
        if self.STEP == "certitude": # For certitude, we will only analyse the toxicite_score to save tokens
            prev_steps_possible_columns = ["toxicite_score"]
        elif self.STEP == "conclusion": 
            prev_steps_possible_columns = ["toxicite_score", "certitude"]
        prev_steps = [col for col in self.df_checkpoint.columns
                      if col not in self.df_all.columns and col in prev_steps_possible_columns]

        for _, row in self.df_checkpoint.iterrows():
            msg_id = row["msg_id"]
            if msg_id in json_batch:
                continue

            fields = {"Message original": row["content"]}
            for step in prev_steps:
                prompt_info = PROMPTS_HEADERS[step]
                fields[prompt_info.name] = row[step]

            json_batch[msg_id] = {
                "custom_id": f"{self.STEP}_{msg_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": create_batch_file(
                    MODEL=self.MODEL,
                    prompt=PROMPTS_HEADERS[self.STEP],
                    fields=fields,
                    max_tokens=self.max_tokens
                )
            }

        self.batch_path.parent.mkdir(parents=True, exist_ok=True)
        with self.batch_path.open("w", encoding="utf-8") as f:
            for entry in json_batch.values():
                f.write(json.dumps(entry) + "\n")
        print(f"Batch file written to {self.batch_path}")

    def upload_batch(self) -> None:
        # Avoid double use of live context from outer script
        file = open(self.batch_path, "rb")
        uploaded = openai.files.create(file=file, purpose="batch")
        self.uploaded_file_id = uploaded.id
        self.console.print(f"[green]✔ Uploaded batch file. ID: [bold]{self.uploaded_file_id}[/bold]")

    def submit_batch(self) -> None:
        if not self.uploaded_file_id:
            raise ValueError("Batch file not uploaded. Call upload_batch() first.")
        self.batch = openai.batches.create(
            input_file_id=self.uploaded_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        self.console.print(f"[green]✔ Batch submitted. ID: [bold]{self.batch.id}[/bold]")

    def wait_for_completion(self) -> Any:
        assert self.batch is not None, "No batch submitted."
        batch_id = self.batch.id
        elapsed = 0
        while True:
            batch = openai.batches.retrieve(batch_id)
            self.console.print(f"[yellow] Batch status (after {elapsed}s): [bold]{batch.status}[/bold]")
            if batch.status in ["completed", "failed", "expired"]:
                return batch
            time.sleep(60)
            elapsed += 60

    def download_and_parse_results(self) -> list[dict[str, Any]]:
        batch = self.wait_for_completion()
        if batch.status != "completed":
            raise RuntimeError(f"Batch did not complete: {batch.status}")

        output_id = batch.output_file_id
        content = openai.files.content(output_id).read()
        self.output_path.write_bytes(content)

        results: list[dict[str, Any]] = []
        for line in content.decode("utf-8").splitlines():
            data = json.loads(line)
            if data.get("error"):
                self.console.print(f"[red] Error for {data['custom_id']}: {data['error']}")
                continue
            msg_id = data["custom_id"].split("_", 1)[1]
            choice = data["response"]["body"]["choices"][0]["message"]["content"].strip()
            usage = data["response"]["body"].get("usage", {})
            results.append({
                "msg_id": msg_id,
                self.STEP: choice,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            })
        return results

    def merge_and_save(self, results: list[dict[str, Any]]) -> None:
        df_results = pd.DataFrame(results)
        df_checkpoint = pd.read_csv(self.CHECKPOINT_CSV, dtype={"msg_id": str})

        df_results["msg_id"] = df_results["msg_id"].str.replace(f"{self.STEP.split('_')[-1]}_", "", regex=False)

        df_updated = df_checkpoint.merge(
            df_results[["msg_id", self.STEP]],
            on="msg_id",
            how="left",
            suffixes=("_old", "")
        )

        old_col = f"{self.STEP}_old"
        if old_col in df_updated.columns:
            df_updated[self.STEP] = df_updated[self.STEP].replace("", np.nan)
            df_updated[self.STEP] = df_updated[self.STEP].combine_first(df_updated[old_col])
            df_updated.drop(columns=[old_col], inplace=True)

        df_updated.to_csv(self.CHECKPOINT_CSV, index=False)
        self.console.print(f"[green]✔ Updated annotations saved to: [bold]{self.CHECKPOINT_CSV}[/bold]")

    def summarize_tokens(self, results: list[dict[str, Any]]) -> None:
        total_prompt = sum(r["prompt_tokens"] for r in results)
        total_comp = sum(r["completion_tokens"] for r in results)
        total_tokens = total_prompt + total_comp
        cost = (total_prompt / 1000) * PRICE_PER_K_PROMPT + (total_comp / 1000) * PRICE_PER_K_COMPLETION

        panel = Panel.fit(
            f"[yellow]Prompt tokens[/yellow]     : {total_prompt}\n"
            f"[yellow]Completion tokens[/yellow] : {total_comp}\n"
            f"[yellow]Total tokens[/yellow]      : {total_tokens}\n"
            f"[bold green]Estimated cost[/bold green]    : [bold]${cost:.4f}[/bold]",
            title="[bold cyan]Token Summary", border_style="cyan")

        self.console.print(panel)

    def run_pipeline(self) -> None:
        self.upload_batch()
        self.submit_batch()
        results = self.download_and_parse_results()
        if results:
            self.merge_and_save(results)
            self.summarize_tokens(results)
        else:
            self.console.print("[red]⚠ No results to merge.")
