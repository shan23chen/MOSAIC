#!/bin/bash

# Base input and output directories
BASE_SAVE_DIR="./output/activations"
BASE_CLASSIFY_DIR="./output/classifications"

BATCH_SIZE=1
MODEL_TYPE="llm"
SAE_LOCATION="res"
TEST_SIZE=0.2
TREE_DEPTH=5
ACT_ONLY="True"

# Set GPU for processing
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0

# Function to extract all tokens using step1_extract_all.py
run_extraction() {
    model_name=$1
    layer=$2
    width=$3
    dataset_name=$4
    dataset_config_name=$5
    dataset_split=$6
    text_field=$7
    label_field=$8

    echo "==============================================="
    echo "Starting token extraction with configuration:"
    echo "Model: ${model_name}"
    echo "Layer: ${layer}"
    echo "Width: ${width}"
    echo "Dataset: ${dataset_name}"
    echo "Split: ${dataset_split}"
    echo "Text Field: ${text_field}"
    echo "Label Field: ${label_field}"
    echo "Save Directory: ${save_dir}"
    echo "==============================================="

    python step1_extract_all.py \
        --model_name ${model_name} \
        --model_type ${MODEL_TYPE} \
        --sae_location ${SAE_LOCATION} \
        --layer ${layer} \
        --save_dir ${BASE_SAVE_DIR} \
        --dataset_name ${dataset_name} \
        --dataset_config_name ${dataset_config_name} \
        --dataset_split ${dataset_split} \
        --text_field ${text_field} \
        --batch_size 2 \
        --image_field NA \
        --label_field ${label_field} \
        --act_only ${ACT_ONLY} \
        --width ${width} \
        --all_tokens True

    status=$?
    if [ ${status} -eq 0 ]; then
        echo "Successfully completed extraction for ${dataset_name} with ${model_name} (width=${width})"
    else
        echo "Error during extraction for ${dataset_name} with ${model_name} (width=${width})"
        echo "Error code: ${status}"
    fi
}

# Function to run dataset classification with step2_dataset_classify.py
run_classification() {
    input_dir=$1
    model_name=$2
    dataset_name=$3
    dataset_config_name=$4
    layers=$5
    width=$6
    dataset_split=$7
    top_n=$8
    binarize_value=$9

    echo "==============================================="
    echo "Starting classification with configuration:"
    echo "Model: ${model_name}"
    echo "Layers: ${layers}"
    echo "Width: ${width}"
    echo "Dataset: ${dataset_name}"
    echo "Split: ${dataset_split}"
    echo "Top N: ${top_n}"
    echo "Binarize Value: ${binarize_value}"
    echo "==============================================="

    python step2_dataset_classify.py \
        --input-dir ${BASE_SAVE_DIR} \
        --dashboard-dir ${BASE_CLASSIFY_DIR} \
        --model-name ${model_name} \
        --dataset-name ${dataset_name} \
        --dataset-config-name ${dataset_config_name} \
        --model-type ${MODEL_TYPE} \
        --dataset-split ${dataset_split} \
        --layer ${layers} \
        --sae_location ${SAE_LOCATION} \
        --width ${width} \
        --top-n ${top_n} \
        --binarize-value ${binarize_value} \
        --test-size ${TEST_SIZE} \
        --tree-depth ${TREE_DEPTH} \
        --save-plots

    status=$?
    if [ ${status} -eq 0 ]; then
        echo "Successfully completed classification for ${dataset_name} with ${model_name} (width=${width})"
    else
        echo "Error during classification for ${dataset_name} with ${model_name} (width=${width})"
        echo "Error code: ${status}"
    fi
}

# Models and configurations
declare -A MODEL_LAYERS
MODEL_LAYERS["google/gemma-2b"]="6,12,17"
MODEL_LAYERS["google/gemma-2-2b"]="5,12,19"
MODEL_LAYERS["google/gemma-2-9b"]="9,20,31"
MODEL_LAYERS["google/gemma-2-9b-it"]="9,20,31"

declare -A MODEL_WIDTHS
MODEL_WIDTHS["google/gemma-2b"]="16k"
MODEL_WIDTHS["google/gemma-2-2b"]="16k 65k"
MODEL_WIDTHS["google/gemma-2-9b"]="16k"
MODEL_WIDTHS["google/gemma-2-9b-it"]="16k"

# Dataset configurations
declare -A DATASETS
# DATASETS["sorry-bench/sorry-bench-202406:train:None"]="turns category"
# DATASETS["Anthropic/election_questions:test:None"]="question label"
# DATASETS["textdetox/multilingual_toxicity_dataset:en:None"]="text toxic"
# DATASETS["AIM-Harvard/reject_prompts:train:None"]="text label"
# DATASETS["jackhhao/jailbreak-classification:test:None"]="prompt type"
DATASETS["cardiffnlp/tweet_sentiment_multilingual:test:english"]="text label"
DATASETS["cardiffnlp/tweet_sentiment_multilingual:test:spanish"]="text label"
DATASETS["cardiffnlp/tweet_sentiment_multilingual:test:french"]="text label"
DATASETS["cardiffnlp/tweet_sentiment_multilingual:test:german"]="text label"
DATASETS["cardiffnlp/tweet_sentiment_multilingual:test:portuguese"]="text label"
DATASETS["cardiffnlp/tweet_sentiment_multilingual:test:italian"]="text label"
DATASETS["cardiffnlp/tweet_sentiment_multilingual:test:arabic"]="text label"
DATASETS["cardiffnlp/tweet_sentiment_multilingual:test:hindi"]="text label"

# Process each model, width, and dataset
for model_name in "${!MODEL_LAYERS[@]}"; do
    layers=${MODEL_LAYERS[$model_name]}
    model_short_name=$(echo ${model_name} | cut -d'/' -f2)

    for width in ${MODEL_WIDTHS[$model_name]}; do
        for dataset in "${!DATASETS[@]}"; do
            dataset_name=$(echo $dataset | cut -d':' -f1)
            dataset_split=$(echo $dataset | cut -d':' -f2)
            dataset_config_name=$(echo $dataset | cut -d':' -f3)
            text_field=$(echo ${DATASETS[$dataset]} | cut -d' ' -f1)
            label_field=$(echo ${DATASETS[$dataset]} | cut -d' ' -f2)

            # Run token extraction
            run_extraction ${model_name} ${layers} ${width} ${dataset_name} ${dataset_split} ${dataset_config_name} ${text_field} ${label_field} ${save_dir}
            
            
            # # Run classification
            for top_n in 0 20 50; do
                run_classification ${save_dir} ${model_name} ${dataset_name} ${dataset_config_name} ${layers} ${width} ${dataset_split} ${top_n} None
                run_classification ${save_dir} ${model_name} ${dataset_name} ${dataset_config_name} ${layers} ${width} ${dataset_split} ${top_n} 1.0
            done
            # Run classification for top_n=-1 and binarize_value=None
            run_classification ${save_dir} ${model_name} ${dataset_name} ${dataset_config_name} ${layers} ${width} ${dataset_split} -1 None
        done
    done
done

echo "All processes completed at: $(date)"
