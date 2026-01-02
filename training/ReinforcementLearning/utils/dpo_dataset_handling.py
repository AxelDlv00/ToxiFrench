import os
from datasets import load_dataset
from functools import partial
from datasets import concatenate_datasets
import numpy as np

class DpoDatasetPreparer:
    def __init__(self, tokenizer, text_field="content", chosen_field="chosen", rejected_field="rejected", label_field="literal_conclusion_annotator", seed=42, oversampling=False):
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.chosen_field = chosen_field
        self.rejected_field = rejected_field
        self.label_field = label_field
        self.seed = seed
        self.oversampling = oversampling

    def _dpo_formatting_func(self, row):
        prompt = f"Message:\n{row[self.text_field].strip()}\n\nAnalyse:\n"
        
        return {
            "prompt": prompt, 
            "chosen": row[self.chosen_field].strip(), 
            "rejected": row[self.rejected_field].strip()
        }

    def prepare(self, dataset_path, split_name="dpo_train"):
        ds = load_dataset(dataset_path, split=split_name)

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
        
        formatted_ds = ds.map(
            self.formatting_func,
            remove_columns=ds.column_names, 
            desc="Formatting dataset"
        )

        return formatted_ds