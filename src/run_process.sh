export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0


python step2_dataset_classify.py \
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


python step2_dataset_classify.py \
    --input-dir ./output_llm_both \
    --dashboard-dir ../dashboard_data \
    --model-name google/gemma-2-2b \
    --checkpoint google/gemma-2-2b \
    --model-type llm \
    --dataset-name textdetox/multilingual_toxicity_dataset \
    --dataset-split en \
    --layer 5,12 \
    --sae_location res \
    --width 16k \
    --top-n 5 \
    --test-size 0.2 \
    --tree-depth 5 \
    --save-plots

python step2_dataset_classify.py \
    --input-dir ./output_llm_both \
    --dashboard-dir ../dashboard_data \
    --model-name google/gemma-2-2b \
    --checkpoint google/gemma-2-2b \
    --model-type llm \
    --dataset-name textdetox/multilingual_toxicity_dataset \
    --dataset-split zh \
    --layer 5,12 \
    --sae_location res \
    --width 16k \
    --top-n 5 \
    --test-size 0.2 \
    --tree-depth 5 \
    --save-plots

