python -m gtnm.utils merge \
    --kwargs \
        ckpt="outputs_2of4/qwen2.5-14b/checkpoint-2000" \
        save_path="2of4_models/qwen2.5-14b" \
        group_size=1 \
        N=2 \
        M=4
