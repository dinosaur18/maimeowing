import torch
import torch.nn.functional as F
from trl import SFTTrainer


def compute_per_example_losses(per_token_losses, position_ids):
    # currrently support only flash attention

    flat_losses = per_token_losses.view(-1)
    flat_pos = position_ids.view(-1)

    is_start = (flat_pos == 0).long()
    seq_ids = is_start.cumsum(dim=0) - 1
    num_examples = seq_ids[-1].item() + 1
    
    example_loss_sums = torch.zeros(
        num_examples, 
        device=flat_losses.device, 
        dtype=flat_losses.dtype
    )
    
    example_loss_sums.scatter_add_(0, seq_ids, flat_losses)
    
    token_counts = torch.bincount(seq_ids, minlength=num_examples).float()
    token_counts = token_counts.clamp(min=1.0)
    
    per_example_means = example_loss_sums / token_counts
    return per_example_means


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
        per_example_losses = compute_per_example_losses(
            per_token_losses, 
            inputs["position_ids"]
        )
        
        base_loss = per_token_losses.mean()

        # compute batch-robustness regularization
        loss_diff = (per_example_losses.unsqueeze(-1) - per_example_losses).abs()
        loss_diff_unique = torch.triu(loss_diff, diagonal=1)
        reg = loss_diff_unique[loss_diff_unique != 0].mean()

        total_loss = base_loss + self.lambda_reg * reg

        print("\n"*4, loss_diff_unique)
        print(per_token_losses.shape)
        print(per_example_losses.shape)
        print("\n"*4, loss_diff_unique[loss_diff_unique != 0], "\n"*4)
        exit()

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
