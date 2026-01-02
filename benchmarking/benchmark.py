import json
import os
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from tqdm.rich import tqdm
from typing import Dict, Any, List
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

from sklearn.metrics import classification_report, roc_auc_score
from rich.console import Console
from rich.table import Table

from utils.toxicity_predictor import ToxicityPredictor 

from utils.models.openai import GPTPredictor
from utils.models.deepseek import DeepseekPredictor
from utils.models.mistral_api import MistralAPIPredictor
from utils.models.mistral_moderation import MistralModerationPredictor
from utils.models.omni import OpenAIModerationPredictor
from utils.models.perspective import PerspectiveAPIPredictor
from utils.models.gemini import GeminiPredictor

from utils.models.qwen25 import Qwen25Predictor
from utils.models.qwen3 import Qwen3Predictor
from utils.models.mistral_local import MistralPredictor
from utils.models.llama_guard import LlamaGuardPredictor
from utils.models.shieldgemma import ShieldGemmaPredictor

from utils.models.camembert import CamemBertPredictor
from utils.models.distilbert import DistilBertPredictor
from utils.models.polyguard import PolyGuardPredictor
from utils.models.roberta import ToxicBertPredictor

from utils.models.toxifrench import ToxiFrenchPredictor

PREDICTOR_MAP = {
    "GPTPredictor": GPTPredictor,
    "DeepseekPredictor": DeepseekPredictor,
    "MistralAPIPredictor": MistralAPIPredictor,
    "MistralModerationPredictor": MistralModerationPredictor,
    "OpenAIModerationPredictor": OpenAIModerationPredictor,
    "PerspectiveAPIPredictor": PerspectiveAPIPredictor,
    "GeminiPredictor": GeminiPredictor,
    "Qwen25Predictor": Qwen25Predictor,
    "Qwen3Predictor": Qwen3Predictor,
    "MistralPredictor": MistralPredictor,
    "LlamaGuardPredictor": LlamaGuardPredictor,
    "ShieldGemmaPredictor": ShieldGemmaPredictor,
    "CamemBertPredictor": CamemBertPredictor,
    "DistilBertPredictor": DistilBertPredictor,
    "PolyGuardPredictor": PolyGuardPredictor,
    "ToxicBertPredictor": ToxicBertPredictor,
    "ToxiFrenchPredictor": ToxiFrenchPredictor,
}

console = Console()

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def insert_examples(prompt, examples_toxic, examples_non_toxic):
    """
    Insert examples into the prompt.
    """
    def prompt_format(text, answer):
        return f"Message :\n« {text} »\nCe message est-il toxique ?\n{answer}"
    toxic_examples = [prompt_format(ex, 'toxique') for ex in examples_toxic]
    non_toxic_examples = [prompt_format(ex, 'non-toxique') for ex in examples_non_toxic]
    examples = toxic_examples + non_toxic_examples
    random.shuffle(examples)
    return prompt.replace('<insert here>', '\n\n'.join(examples))

def config_experiment(exp_name: str, config: Dict[str, Any]):
    assert exp_name in config.keys(), f"Experiment '{exp_name}' not found in configuration."

    load_dotenv(dotenv_path=config['global']['env_dir'],override=True) 
    
    experiment_config = config[exp_name].copy()
    experiment_config['experiment_name'] = exp_name
    experiment_config['benchmark'] = Path(config["global"]["benchmark_dir"]) / config[exp_name].get("benchmark")
    experiment_config['output'] = Path(config["global"]["results_dir"]) / config[exp_name].get("output")

    if "api_key" in config[exp_name]:
        api_key_config = config[exp_name]["api_key"]
        if api_key_config["type"] == "env":
            api_key = os.getenv(api_key_config["name"])
        elif api_key_config["type"] == "file":
            with open(api_key_config["path"], 'r') as f:
                api_key = f.read().strip()
        else:
            raise ValueError("Unsupported API key configuration type.")
        
        experiment_config['api_key'] = api_key
            
    if "system_prompt" in config[exp_name]:
        system_prompt_config = config[exp_name]["system_prompt"]
        if system_prompt_config["type"] == "file":
            with open(system_prompt_config["path"], 'r') as f:
                system_prompt = f.read().strip()
        else:
            raise ValueError("Unsupported system prompt configuration type.")
        
        experiment_config['system_prompt'] = system_prompt

    if "prompt_template" in config[exp_name]:
        prompt_template_config = config[exp_name]["prompt_template"]
        if prompt_template_config["type"] == "file":
            with open(prompt_template_config["path"], 'r') as f:
                prompt_template = f.read().strip()
        else:
            raise ValueError("Unsupported prompt template configuration type.")
        if config[exp_name].get("few_shots_toxic", 0) + config[exp_name].get("few_shots_non_toxic", 0) > 0:
            few_shots_toxic = config[exp_name].get("few_shots_toxic", 0)
            few_shots_non_toxic = config[exp_name].get("few_shots_non_toxic", 0)
            examples_toxic = config["global"]["few_shot_examples"]["toxic"][:few_shots_toxic]
            examples_non_toxic = config["global"]["few_shot_examples"]["non_toxic"][:few_shots_non_toxic]
            prompt_template = insert_examples(prompt_template, examples_toxic, examples_non_toxic)
        experiment_config['prompt_template'] = prompt_template

    default_gen_params = {
        "max_new_tokens": 1024,
        "do_sample": True,  
        "temperature": 0.7,
        "top_p": 1.0,
        "repetition_penalty": 1.1
    }
    
    if "generation_params" in config[exp_name]:
        user_gen_params = config[exp_name]["generation_params"]
        experiment_config['generation_params'] = {**default_gen_params, **user_gen_params}
    else:
        experiment_config['generation_params'] = default_gen_params
    
    experiment_config['language'] = config[exp_name].get("language", "fr")
    experiment_config['max_retries'] = config[exp_name].get("max_retries", 5)
    experiment_config['delay_base'] = config[exp_name].get("delay_base", 2)
    experiment_config['parallel_requests'] = config[exp_name].get("parallel_requests", 4)
    return experiment_config


def load_benchmark(exp_config: Dict[str, Any]) -> pd.DataFrame:
    benchmark_path = exp_config['benchmark']
    df = pd.read_csv(benchmark_path, encoding="utf-8")
    df = df.dropna(subset=["content", "label"])
    df["label"] = df["label"].astype(int)
    return df

def run_predictions(exp_config: Dict[str, Any], df: pd.DataFrame, result_dir: Path = Path("benchmarking/results")):
    predictor = PREDICTOR_MAP[exp_config['predictor']](exp_config)
    predictor.initialise_predictor()

    output = exp_config['output']
    content_field = exp_config.get("content_field", "content")

    if not (result_dir / output).exists():
        result_dir.mkdir(parents=True, exist_ok=True)
        texts = df[content_field].tolist()
        results = [None] * len(texts)
    else:
        existing_df = pd.read_csv(result_dir / output, encoding="utf-8")
        existing_texts = existing_df[content_field].tolist()
        texts = df[~df[content_field].isin(existing_texts)][content_field].tolist()
        results = existing_df["label"].tolist() + [None] * len(texts)

    with ThreadPoolExecutor(max_workers=exp_config['parallel_requests']) as executor:
        future_to_idx = {
            executor.submit(predictor.predict, txt): idx
            for idx, txt in enumerate(texts)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(texts)):
            idx = future_to_idx[future]
            results[idx] = future.result()
            if idx % 50 == 0:
                pd.DataFrame({"content": texts, "label": results}).to_csv(output, index=False)
                print(f"Checkpoint saved at {output}")
    
    pd.DataFrame({"content": texts, "label": results}).to_csv(output, index=False)
    print(f"Final results saved at {output}")

def run_predictions_batched(exp_config, df, result_dir: Path, batch_size=8):
    predictor = PREDICTOR_MAP[exp_config['predictor']](exp_config)
    predictor.initialise_predictor()
    
    output_filename = Path(exp_config['output']).name 
    output_path = result_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content_field = exp_config.get("content_field", "content")
    
    texts = df[content_field].tolist()
    results = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Batch Inference"):
        batch = texts[i:i + batch_size]
        batch_results = predictor.predict_batch(batch) 
        results.extend(batch_results)
        
        if i % (batch_size * 10) == 0:
            pd.DataFrame({
                "content": texts[:len(results)], 
                "label": results
            }).to_csv(output_path, index=False)

    pd.DataFrame({"content": texts, "label": results}).to_csv(output_path, index=False)
    print(f"Final results saved at {output_path}")

def evaluate_predictions(exp_config: Dict[str, Any]):
    """
    Calcule les métriques (Precision, Recall, F1, Accuracy, AUC) 
    en comparant le benchmark (Ground Truth) et le fichier de sortie (Predictions).
    Sauvegarde un résumé CSV.
    """
    benchmark_path = exp_config['benchmark']
    output_path = exp_config['output']
    exp_name = exp_config.get('experiment_name', 'Unknown')

    if not output_path.exists():
        console.print(f"[red]Output file {output_path} not found. Cannot evaluate.[/red]")
        return

    df_true = pd.read_csv(benchmark_path, encoding="utf-8")
    df_pred = pd.read_csv(output_path, encoding="utf-8")
    content_field = exp_config.get("content_field", "content")

    df_true = df_true[[content_field, 'label']].rename(columns={'label': 'target'})
    df_pred = df_pred[['content', 'label']].rename(columns={'label': 'prediction', 'content': content_field})

    merged_df = pd.merge(df_true, df_pred, on=content_field, how='inner')
    
    if len(merged_df) == 0:
        console.print("[red]No matching content found between benchmark and predictions.[/red]")
        return

    def clean_label(val):
        if isinstance(val, (int, float)):
            return int(val)
        s = str(val).lower().strip()
        if s in ['oui', 'toxic', 'true', '1']:
            return 1
        if s in ['non', 'non-toxic', 'false', '0']:
            return 0
        return 0 

    y_true = merged_df['target'].apply(clean_label)
    y_pred = merged_df['prediction'].apply(clean_label)

    results_row = {"Model": exp_name}
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    results_row.update({
        "Precision_0": report['0']['precision'],
        "Recall_0":    report['0']['recall'],
        "F1_0":        report['0']['f1-score'],
        "Precision_1": report['1']['precision'],
        "Recall_1":    report['1']['recall'],
        "F1_1":        report['1']['f1-score'],
        "Accuracy":    report['accuracy'],
    })

    try:
        roc = roc_auc_score(y_true, y_pred)
        results_row["ROC_AUC"] = roc
    except Exception:
        results_row["ROC_AUC"] = 0.0

    table = Table(title=f"Results for {exp_name}")
    for key in results_row.keys():
        table.add_column(key, justify="center")
    
    row_values = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in results_row.values()]
    table.add_row(*row_values)
    console.print(table)

    summary_path = output_path.parent / "metrics_summary.csv"
    
    new_row_df = pd.DataFrame([results_row])
    
    if summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        existing_summary = existing_summary[existing_summary["Model"] != exp_name]
        final_summary = pd.concat([existing_summary, new_row_df], ignore_index=True)
    else:
        final_summary = new_row_df
        
    final_summary.to_csv(summary_path, index=False)
    console.print(f"[green]Metrics saved to {summary_path}[/green]")

if __name__ == "__main__":
    # Run with : python benchmark.py --config config.json --experiment experiment_name
    parser = argparse.ArgumentParser(description="Run toxicity prediction benchmarks.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file.")
    parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment to run.")
    parser.add_argument("--from_csv", type=bool, default=False, help="If True, skip predictions and only evaluate existing results.")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_config = config_experiment(args.experiment, config)
    result_dir = config["global"]["results_dir"]
    df = load_benchmark(exp_config)
    if not args.from_csv:
        console.print(f"[blue]Running predictions for experiment '{args.experiment}'...[/blue]")
        if exp_config.get("gpu_parallel", False):
            run_predictions_batched(exp_config, df, batch_size=exp_config.get("batch_size", 8), result_dir=Path(result_dir))
        else:
            run_predictions(exp_config, df, result_dir=Path(result_dir))
    evaluate_predictions(exp_config)