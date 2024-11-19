#!/bin/bash

# Base input and output directories
BASE_INPUT_DIR="./outputs"
BASE_SAVE_DIR="./output_llm"
DASHBOARD_DIR="../dashboard_data"
MODEL_TYPE="llm"
SAE_LOCATION="res"
TOP_N=0
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
    dataset_split=$5
    text_field=$6
    label_field=$7
    save_dir=$8

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
        --save_dir ${save_dir} \
        --dataset_name ${dataset_name} \
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
    layers=$4
    width=$5
    dataset_split=$6

    echo "==============================================="
    echo "Starting classification with configuration:"
    echo "Model: ${model_name}"
    echo "Layers: ${layers}"
    echo "Width: ${width}"
    echo "Dataset: ${dataset_name}"
    echo "Split: ${dataset_split}"
    echo "==============================================="

    python step2_dataset_classify.py \
        --input-dir ${input_dir} \
        --dashboard-dir ${DASHBOARD_DIR} \
        --model-name ${model_name} \
        --dataset-name ${dataset_name} \
        --model-type ${MODEL_TYPE} \
        --dataset-split ${dataset_split} \
        --layer ${layers} \
        --sae_location ${SAE_LOCATION} \
        --width ${width} \
        --top-n ${TOP_N} \
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
MODEL_WIDTHS["google/gemma-2-9b"]="16k 131k"
MODEL_WIDTHS["google/gemma-2-9b-it"]="16k 131k"

# Dataset configurations
declare -A DATASETS
DATASETS["sorry-bench/sorry-bench-202406:train"]="turns category"
DATASETS["Anthropic/election_questions:test"]="question label"
DATASETS["textdetox/multilingual_toxicity_dataset:en"]="text toxic"
DATASETS["AIM-Harvard/reject_prompts:train"]="text label"
DATASETS["jackhhao/jailbreak-classification:test"]="prompt type"

# Process each model, width, and dataset
for model_name in "${!MODEL_LAYERS[@]}"; do
    layers=${MODEL_LAYERS[$model_name]}
    model_short_name=$(echo ${model_name} | cut -d'/' -f2)

    for width in ${MODEL_WIDTHS[$model_name]}; do
        for dataset in "${!DATASETS[@]}"; do
            dataset_name=$(echo $dataset | cut -d':' -f1)
            dataset_split=$(echo $dataset | cut -d':' -f2)
            text_field=$(echo ${DATASETS[$dataset]} | cut -d' ' -f1)
            label_field=$(echo ${DATASETS[$dataset]} | cut -d' ' -f2)
            save_dir="${BASE_SAVE_DIR}/${model_short_name}/width_${width}/${dataset_name}_${dataset_split}"
            mkdir -p ${save_dir}

            # Run token extraction
            run_extraction ${model_name} ${layers} ${width} ${dataset_name} ${dataset_split} ${text_field} ${label_field} ${save_dir}

            # # Run classification
            # input_dir="${BASE_INPUT_DIR}/${model_short_name}/width_${width}"
            # run_classification ${input_dir} ${model_name} ${dataset_name} ${layers} ${width} ${dataset_split}
        done
    done
done

echo "All processes completed at: $(date)"
