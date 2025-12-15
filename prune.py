import torch
from dataclasses import dataclass
from trl import SFTConfig, SFTTrainer, TrlParser
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import load_dataset

# from gtnm.baro import BaRoSFTTrainer as SFTTrainer
from gtnm import NAME_TO_NM_MAKER
from gtnm.callbacks import AnnealingCallback


@dataclass
class DataArguments:
    dataset_name: str = None
    dataset_config_name: str = None
    train_files: str = None
    streaming: bool = True


@dataclass
class ModelArguments:
    model_name_or_path: str = None
    dtype: str = "auto"
    attn_implementation: str = "flash_attention_2"
    
    # N:M settings
    group_size: int = 1
    N: int = 2
    M: int = 4
    hard: bool = False
    init_std: float = 1e-2
    eps: float = 1e-6
    tau_init: float = 1.0
    tau_min: float = 0.05
    logit_scaling_init: float = 1.0
    logit_scaling_max: float = 500.0
    

def main():
    parser = TrlParser([SFTConfig, ModelArguments, DataArguments])
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    training_args.accelerator_config.dispatch_batches = False
    set_seed(training_args.seed)
    
    train_dataset = load_dataset(
        path=data_args.dataset_name,
        name=data_args.dataset_config_name,
        data_files=data_args.train_files,
        streaming=data_args.streaming,
        split="train"
    ).shuffle(seed=training_args.data_seed, buffer_size=50000)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=model_args.dtype,
        attn_implementation=model_args.attn_implementation
    )
    make_nm_func_ = NAME_TO_NM_MAKER[model.config.model_type] 
    model = make_nm_func_(
        model=model,
        group_size=model_args.group_size,
        N=model_args.N,
        M=model_args.M,
        freeze_model=True,
        hard=model_args.hard,
        init_std=model_args.init_std,
        eps=model_args.eps,
        tau=model_args.tau_init,
        logit_scaling=model_args.logit_scaling_init
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer, 
        callbacks=[
            AnnealingCallback(
                logit_scaling_init=model_args.logit_scaling_init,
                logit_scaling_max=model_args.logit_scaling_max,
                tau_init=model_args.tau_init,
                tau_min=model_args.tau_min,
                num_steps=training_args.max_steps
            )
        ]
    )
        
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint 
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    

if __name__ == "__main__":
    main()
