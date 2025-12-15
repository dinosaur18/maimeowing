python -m gtnm.utils merge \
    --kwargs \
        ckpt="outputs/gemma3-1b/checkpoint-2000" \
        save_path="nm_models/gemma3-1b" \
        group_size=1 \
        N=2 \
        M=4
