from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from huggingface_hub import load_torch_model
from argparse import ArgumentParser
import logging

from .mask import GTNMMask
from . import NAME_TO_NM_MAKER


logger = logging.getLogger(__name__)


def merge_masks_(model: nn.Module):
    """Merge N:M masks into weights and remove mask modules"""
    
    model.eval()
    
    for name, module in model.named_modules():
        if hasattr(module, "mask") and isinstance(module.mask, GTNMMask):
            logger.info(f"Merging {name}...")

            w = module.weight
            w.data = module.mask.apply_to_weight(w.data)
            
            del module.mask
            
            # Replace the N:M module's forward method by the default one
            module.forward = module.__class__.forward.__get__(module)
    return model

    
def merge_nm_checkpoint(ckpt, save_path, group_size, N=2, M=4):
    """Produce standard HuggingFace models from N:M checkpoints"""
    
    logger.info("Initializing model...")
    config = AutoConfig.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    make_nm_func_ = NAME_TO_NM_MAKER[model.config.model_type]
    model = make_nm_func_(model, group_size=group_size, N=N, M=M)
    
    logger.info("Loading N:M checkpoint...")
    load_torch_model(model, ckpt)
    
    model = merge_masks_(model)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    

if __name__ == "__main__":
    def parse_cmd_kwargs(pairs: str):
        result = {}
        for pair in pairs:
            key, value = pair.split('=')
            try:
                value = eval(value)
            except:
                pass
            result[key] = value
        return result
    
    TASK_TO_FUNC = {
        "merge": merge_nm_checkpoint
    }
    
    parser = ArgumentParser()
    
    parser.add_argument("task")
    parser.add_argument("--kwargs", nargs="*")
    
    args = parser.parse_args()
    
    func = TASK_TO_FUNC[args.task]
    func(**parse_cmd_kwargs(args.kwargs))