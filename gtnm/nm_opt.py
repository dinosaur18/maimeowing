from torch import nn
from transformers import OPTForCausalLM
import logging

from .functions import make_nm_linear_


logger = logging.getLogger(__name__)


def make_nm_opt_(model: OPTForCausalLM, group_size, N=2, M=4, freeze_model=False, **mask_init_kwargs):
    for param in model.parameters():
        param.requires_grad = not freeze_model
    
    logger.info(f"Sparsifying modules...")
    
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Linear):
            make_nm_linear_(module, group_size, N, M, **mask_init_kwargs)
    
    if model.model.decoder.embed_tokens.weight is not model.lm_head.weight:
        logger.info("Sparsifying lm_head...")
        make_nm_linear_(model.lm_head, group_size, N, M, **mask_init_kwargs)
    
    return model