import sys
import logging

from .mask import GTNMMask
from .nm_llama import make_nm_llama_
from .nm_opt import make_nm_opt_


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s|%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

NAME_TO_NM_MAKER = {
    "llama": make_nm_llama_,
    "qwen2": make_nm_llama_,
    "opt": make_nm_opt_,
    "gemma3": make_nm_llama_,
    "gemma3_text": make_nm_llama_
}

__all__ = [
    "GTNMMask",
    "NAME_TO_NM_MAKER"
]