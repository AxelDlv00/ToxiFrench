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
    system_prompt: str,
    prompt: Any,
    max_tokens: int = 150
) -> Dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }

# ==========================================================
# Class for Handling OpenAI API Responses
# ==========================================================

class OpenAIResponseHandler:

    def __init__(
        self,
        df: pd.DataFrame,
        USE_PROXY: str = "",
        MODEL: str = "gpt-4o-mini",
        MAX_CONTENTS: int | None = None,
        batch_path: Path = Path("batch.jsonl"),
        max_tokens: int = 150,
        path_api_key: Path = Path("path/to/api_key.txt"),
        output_path: Path = Path("output.jsonl"),
    ):
        self.MAX_CONTENTS = min(MAX_CONTENTS, len(df)) if MAX_CONTENTS else len(df)
        self.df = df.sample(n=self.MAX_CONTENTS, random_state=42)
        self.batch_path = batch_path
        self.max_tokens = max_tokens
        self.path_api_key = path_api_key
        self.output_path = output_path
        self.MODEL = MODEL
        self.batch = None
        self.uploaded_file_id = None
        self.console = Console()

        log = ""
        if USE_PROXY:
            log += self.set_proxy(USE_PROXY)
        log += self.init_client()
        self.console.print(Panel(log, title="[bold green]Init", subtitle=""))

    def set_proxy(self, proxy_url: str):
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        return f"Proxy set to: {proxy_url}\n"

    def init_client(self):
        openai.api_key = self.path_api_key.read_text().strip()
        return "OpenAI API key set."

    def create_json_batch(self) -> None:
        json_batch: dict[str, Any] = {}

        for _, row in self.df.iterrows():
            msg_id = row["msg_id"]
            if msg_id in json_batch:
                continue

            json_batch[msg_id] = {
                "custom_id": f"{msg_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": create_batch_file(
                    MODEL=self.MODEL,
                    system_prompt=row["system_prompt"],
                    prompt=row["prompt"],
                    max_tokens=self.max_tokens
                )
            }

        self.batch_path.parent.mkdir(parents=True, exist_ok=True)
        with self.batch_path.open("w", encoding="utf-8") as f:
            for entry in json_batch.values():
                f.write(json.dumps(entry) + "\n")
        print(f"Batch file written to {self.batch_path}")

    def upload_batch(self) -> None:
        with open(self.batch_path, "rb") as file:
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
            msg_id = data["custom_id"]
            choice = data["response"]["body"]["choices"][0]["message"]["content"].strip()
            usage = data["response"]["body"].get("usage", {})
            results.append({
                "msg_id": msg_id,
                "response": choice,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            })
        return results

    def merge_and_save(self, results: list[dict[str, Any]]) -> None:
        df_results = pd.DataFrame(results)
        df_merged = self.df.merge(df_results, on="msg_id", how="left")
        self.df = df_merged  # Update internal state
        save_path = self.output_path.with_suffix(".merged.csv")
        df_merged.to_csv(save_path, index=False)
        self.console.print(f"[cyan]✔ Results merged and saved to [bold]{save_path}[/bold]")

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
        self.create_json_batch()
        self.upload_batch()
        self.submit_batch()
        results = self.download_and_parse_results()
        if results:
            self.merge_and_save(results)
            self.summarize_tokens(results)
        else:
            self.console.print("[red]⚠ No results to merge.")