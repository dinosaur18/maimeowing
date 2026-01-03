python -m gtnm.utils merge \
    --kwargs \
        ckpt="outputs_2of8/qwen2.5-7b/checkpoint-2000" \
        save_path="2of8_models/qwen2.5-7b" \
        group_size=1 \
        N=2 \
        M=8
