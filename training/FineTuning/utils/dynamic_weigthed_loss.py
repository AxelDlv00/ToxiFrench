import torch
import torch.nn as nn
from trl.trainer.utils import selective_log_softmax
from trl.trainer.sft_trainer import SFTTrainer

def dwl_loss(outputs, labels, class_start_id=None, class_end_id=None, num_items_in_batch=None, alphas=[1.0], DFT=False):
    """
    Implementation of the Dynamic Weighted Loss (DWL) function.
    
    Arguments:
        outputs: Model outputs containing logits [B, L, V]
        labels: Target IDs [B, L]
        num_items_in_batch: Number of tokens for normalization (N)
        class_start_id: Token ID marking the start of a block
        class_end_id: Token ID marking the end of a block
        alphas: List of weights [Out-of-block, Block1, Block2, ...]
        DFT: If True, applies the -sg(p)*log(p) rectification from https://huggingface.co/papers/2508.05629 implemented in https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py 
    """
    logits = outputs.logits  
    labels = nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = shift_labels != -100
    shift_labels[~loss_mask] = 0
    logprobs = selective_log_softmax(logits, shift_labels)
    
    if DFT:
        # -sg(π) * log(π)
        per_token_loss = -logprobs.exp().detach() * logprobs
    else:
        #   -log(π)
        per_token_loss = -logprobs
    
    # Identify the start and end of blocks
    if class_start_id is None or class_end_id is None:
        # Then everything is "out-of-block"
        class_mask = torch.zeros_like(shift_labels, dtype=torch.long, device=labels.device)
    else:
        starts = (shift_labels == class_start_id).long() # [1, 0, ...., 0, 1, 0, ...]
        ends = (shift_labels == class_end_id).long()   # [0, 0, ..., 1, 0, ..., 0, 1]
        starts_cum = starts.cumsum(dim=1) # [1, 1, ..., 1, 2, ..., 2, 3, ...]
        ends_cum = ends.cumsum(dim=1) # [0, 0, ..., 1, 1, ..., 1, 2, ...]
        ends_cum_shifted = ends_cum.roll(shifts=1, dims=1)
        ends_cum_shifted[:, 0] = 0
        in_block_mask = (starts_cum > ends_cum_shifted) 
        class_mask = (starts_cum * in_block_mask.long())

    alphas_mapping = torch.tensor(alphas, device=logits.device, dtype=logits.dtype)
    
    # If there are more blocks than defined alphas, pad with 1.0 (i.e. no weighting)
    max_block_id = class_mask.max().item()
    if max_block_id >= len(alphas_mapping):
        extra_ones = torch.ones(int(max_block_id - len(alphas_mapping) + 1), device=logits.device, dtype=logits.dtype)
        alphas_mapping = torch.cat([alphas_mapping, extra_ones])
    
    # Map the coefficients to each token
    alpha_mask = alphas_mapping[class_mask.long()]
    weighted_loss = alpha_mask * per_token_loss * loss_mask.float()
    
    if num_items_in_batch is None:
        num_items_in_batch = loss_mask.sum()
    return weighted_loss.sum() / num_items_in_batch

class CustomDWLTrainer(SFTTrainer):
    def __init__(self, think_start_id, think_end_id, weight_schedule, DFT, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.weight_schedule = weight_schedule
        self.DFT = DFT

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        labels = inputs.get("labels")
        
        current_epoch = self.state.epoch or 0
        current_alphas = [1.0] # Default
        for entry in self.weight_schedule:
            if current_epoch >= entry["epoch"]:
                current_alphas = entry["alphas"]
    
        loss = dwl_loss(
            outputs=outputs,
            labels=labels,
            class_start_id=self.think_start_id,
            class_end_id=self.think_end_id,
            alphas=current_alphas,
            num_items_in_batch=num_items_in_batch,
            DFT=self.DFT
        )
        
        return (loss, outputs) if return_outputs else loss