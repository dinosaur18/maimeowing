import torch
import torch.nn.functional as F
from trl import SFTTrainer


def compute_batch_robustness_loss(per_token_losses, num_tokens):
    # per_token_losses: (batch_size * seq_len,)
    # take randomly num_tokens losses and minimize their pairwise differences on average
    flat_losses = per_token_losses.view(-1)
    sampled_losses = flat_losses[torch.randperm(flat_losses.size(0))[:num_tokens]]
    loss_diff = (sampled_losses.unsqueeze(-1) - sampled_losses).abs()
    loss_diff_unique = torch.triu(loss_diff, diagonal=1)
    br_loss = loss_diff_unique[loss_diff_unique != 0].mean()
    return br_loss


class BaRoSFTTrainer(SFTTrainer):

    def __init__(self, *args, lambda_reg=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = lambda_reg
        self._metrics["train"]["baro"] = []
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):        
        mode = "train" if self.model.training else "eval"
        
        labels = inputs.pop("labels")
        logits = model(**inputs).logits
        vocab_size = logits.size(-1)
        logits = logits.view(-1, vocab_size)
        
        # compute per-example cross-entropy losses
        labels = F.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].view(-1)
        per_token_losses = F.cross_entropy(logits, shift_labels, reduction="none")
        
        base_loss = per_token_losses.mean()
        reg = compute_batch_robustness_loss(per_token_losses, num_tokens=16384)  # TODO: make num_tokens a parameter

        total_loss = base_loss + self.lambda_reg * reg
        
        if num_items_in_batch is not None:
            total_loss = total_loss / self.args.gradient_accumulation_steps
        
        # log metrics
        if mode == "train":
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        self._metrics[mode]["baro"].append(reg.detach().item())
        
        return total_loss
