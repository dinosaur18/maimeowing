from torch import nn
from transformers import LlamaForCausalLM
import logging

from .functions import make_nm_linear_


logger = logging.getLogger(__name__)


def make_nm_llama_(model: LlamaForCausalLM, group_size, N=2, M=4, freeze_model=True, **mask_init_kwargs):
    for param in model.parameters():
        param.requires_grad = not freeze_model
    
    logger.info(f"Sparsifying modules...")
    
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Linear):
            make_nm_linear_(module, group_size, N, M, **mask_init_kwargs)
    
    return model