export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0


python process_npz_files.py \
    --input-dir ./output_llm_both \
    --dashboard-dir ../dashboard_data \
    --model-name google/gemma-2-2b \
    --checkpoint google/gemma-2-2b \
    --dataset-name Anthropic/election_questions \
    --model-type llm \
    --dataset-split test \
    --model-type llm \
    --layer 5,12 \
    --sae_location res \
    --width 16k \
    --top-n 5 \
    --test-size 0.2 \
    --tree-depth 5 \
    --save-plots

