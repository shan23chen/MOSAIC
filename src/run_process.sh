export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0


# python process_npz_files.py \
#     --input-dir ./output_llm_both \
#     --model-name google/gemma-2b-it \
#     --model-type llm \
#     --layer 12 \
#     --sae-release gemma-2b \
#     --top-n 5 \
#     --output-dir processed_features_llm \
#     --test-size 0.2 \
#     --tree-depth 5 \
#     --save-plots