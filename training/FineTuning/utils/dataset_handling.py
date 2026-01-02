import os
from datasets import load_dataset
from functools import partial
from datasets import concatenate_datasets
import numpy as np
from transformers import AutoTokenizer

class DatasetPreparer:
    def __init__(self, tokenizer, text_field="content", cot_fields=["direct_question"], 
                 label_field="literal_conclusion_annotator", seed=42, 
                 oversampling=False, max_length=1024):
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.label_field = label_field
        self.cot_fields = cot_fields
        self.seed = seed
        self.oversampling = oversampling
        self.max_length = max_length

    def tokenize_function(self, example):
        return self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False, 
        )

    def formatting_func(self, row):
        prompt = f"Message:\n{row[self.text_field].strip()}\n\nAnalyse:\n"
        full_thought = "\n\n".join([row[f].strip() for f in self.cot_fields if f in row and row[f]])
        
        completion = (
            f"{full_thought}\n"  
            f"Conclusion : {row[self.label_field].strip()}\n"
            f"{self.tokenizer.eos_token}"
        )
        
        full_text = prompt + completion
        
        return {
            "text": full_text,
            "prompt": prompt,
            "completion": completion
        }

    def prepare(self, dataset_path, split_name="train", cache=False, remove_columns=["text", "prompt", "completion"]):
        ds = load_dataset(dataset_path, split=split_name, download_mode="force_redownload")

        if self.oversampling:
            labels = ds[self.label_field]
            unique_labels, counts = np.unique(labels, return_counts=True)
            max_count = max(counts)
            
            datasets_to_combine = []
            
            for label in unique_labels:
                label_ds = ds.filter(lambda x: x[self.label_field] == label)
                current_count = len(label_ds)
                if current_count < max_count:
                    repeat_factor = max_count // current_count
                    remainder = max_count % current_count                    
                    duplicated_ds = [label_ds] * repeat_factor
                    if remainder > 0:
                        duplicated_ds.append(label_ds.select(range(remainder)))
                    datasets_to_combine.append(concatenate_datasets(duplicated_ds))
                else:
                    datasets_to_combine.append(label_ds)            
            ds = concatenate_datasets(datasets_to_combine)
            ds = ds.shuffle(seed=self.seed)
        
        ds = ds.map(
            self.formatting_func,
            remove_columns=ds.column_names, 
            desc="Formatting dataset",
            load_from_cache_file=cache
        )

        tokenized_ds = ds.map(
            self.tokenize_function,
            batched=False,
            remove_columns=remove_columns,
            desc="Tokenizing dataset",
            load_from_cache_file=cache
        )

        return tokenized_ds

if __name__ == "__main__":
    # os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
    # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", use_fast=True, trust_remote_code=True)
    preparer = DatasetPreparer(tokenizer, text_field="content", cot_fields=["CoT_explication", "CoT_tons", "CoT_intentions", "CoT_categorie", "CoT_score", "cot_final_question"], label_field="literal_conclusion_annotator", oversampling=True)
    ds_test = preparer.prepare("Naela00/ToxiFrench", split_name="test")
    ds_train = preparer.prepare("Naela00/ToxiFrench", split_name="train")
    for i in range(3):
        sample = ds_test[i]
        print("Prompt:\n", sample["prompt"])
        print('- '*25)
        print("Completion:\n", sample["completion"])
        print("="*50)