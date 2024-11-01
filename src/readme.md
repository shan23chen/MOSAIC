# SAE-LLaVA code Readme

This directory contains the source code for SAE-LLaVA.  See individual file headers for details.

The two main run files are `step1_extract_all.py`, `step2_dataset_classify.py`.

Step 1 is mainly to extract the hidden states, calculate the sae eatures as request:

```bash
python step1_extract_all.py --model_name google/gemma-2-2b \
    --model_type llm \
    --sae_location res \
    --layer 12 \
    --save_dir ./output_llm_both1/ \
    --dataset_name Anthropic/election_questions \
    --dataset_split test \
    --text_field question \
    --batch_size 16 \
    --image_field NA \
    --label_field label \
    --act_only False \
    --width 16k \
    --all_tokens True
```

Step 2 is mainly to the hidden states, sae features into datasets, encode them into features as request. Then train the classifier and render the overview dashboard.

```bash
python step2_dataset_classify.py \
    --input-dir ./output_llm_both1 \
    --dashboard-dir ../dashboard_data \
    --model-name google/gemma-2-2b \
    --dataset-name Anthropic/election_questions \
    --model-type llm \
    --dataset-split test \
    --model-type llm \
    --layer 12 \
    --sae_location res \
    --width 16k \
    --top-n 5 \
    --test-size 0.2 \
    --tree-depth 5 \
    --save-plots
```
