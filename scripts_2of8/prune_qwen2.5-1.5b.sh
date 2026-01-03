export CUDA_VISIBLE_DEVICES=6

IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#GPU_ARRAY[@]}

accelerate launch \
    --config_file configs/single_gpu.yaml \
    --num_processes $NPROC \
    prune.py \
    --model_name_or_path "qwen/qwen2.5-1.5b" \
    --N 2 \
    --M 8 \
    --dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --dataset_name parquet \
    --train_files "/llm-data/other/data/for_susi_v2_enumerated/*.parquet" \
    --streaming true \
    --optim adamw_8bit \
    --learning_rate 1e-3 \
    --warmup_steps 0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --weight_decay 0.05 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 36 \
    --dataset_text_field text \
    --max_length 4096 \
    --report_to tensorboard \
    --run_name susi \
    --packing true \
    --logging_steps 1 \
    --max_steps 2000 \
    --output_dir "outputs_2of8/qwen2.5-1.5b" \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --logit_scaling_init 1.0 \
    --logit_scaling_max 500.0 \
    --seed 59 \
    --data_seed 59 \
    --gradient_checkpointing false
