#!/bin/sh

# Base directories for input and output
BASE_INPUT_DIR="./outputs"
BASE_SAVE_DIR="./output_llm_large"
DASHBOARD_DIR="../dashboard_data"

# Configuration constants
MODEL_TYPE="llm"
SAE_LOCATION="res"
TOP_N=0
TEST_SIZE=0.2
TREE_DEPTH=5

# Set GPU for processing
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0

# Function to run token extraction using step1_extract_all.py
run_extraction() {
    model_name=$1
    layers=$2
    width=$3
    dataset_name=$4
    text_field=$5
    label_field=$6
    dataset_split=$7
    save_dir="${BASE_SAVE_DIR}/${model_name}/width_${width}/${dataset_name}_${dataset_split}"

    echo "==============================================="
    echo "Starting extraction with configuration:"
    echo "Model: ${model_name}"
    echo "Layers: ${layers}"
    echo "Width: ${width}"
    echo "Dataset: ${dataset_name}"
    echo "Text Field: ${text_field}"
    echo "Label Field: ${label_field}"
    echo "Split: ${dataset_split}"
    echo "Save Directory: ${save_dir}"
    echo "==============================================="

    mkdir -p ${save_dir}

    python step1_extract_all.py \
        --model_name ${model_name} \
        --model_type ${MODEL_TYPE} \
        --sae_location ${SAE_LOCATION} \
        --layer ${layers} \
        --save_dir ${save_dir} \
        --dataset_name ${dataset_name} \
        --dataset_split ${dataset_split} \
        --text_field ${text_field} \
        --batch_size 16 \
        --image_field NA \
        --label_field ${label_field} \
        --act_only False \
        --width ${width} \
        --all_tokens True

    status=$?
    if [ ${status} -eq 0 ]; then
        echo "Successfully completed extraction for ${dataset_name} (width=${width}, model=${model_name})"
    else
        echo "Error during extraction for ${dataset_name} (width=${width}, model=${model_name})"
        echo "Error code: ${status}"
    fi
}

# Function to run classification using step2_dataset_classify.py
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
        echo "Successfully completed classification for ${dataset_name} (width=${width}, model=${model_name})"
    else
        echo "Error during classification for ${dataset_name} (width=${width}, model=${model_name})"
        echo "Error code: ${status}"
    fi
}

# Create base output directory
mkdir -p ${BASE_SAVE_DIR}

echo "Starting extraction and classification processes at: $(date)"

# Process missing configurations
echo "Processing missing configurations for larger models and widths..."

# Gemma 2 2B (width 1m)
MODEL_NAME="google/gemma-2-2b"
LAYERS="5,12,19"
WIDTH="1m"
DATASETS=("AIM-Harvard/sorrybench:turns:category:train" \
          "Anthropic/election_questions:question:label:test" \
          "textdetox/multilingual_toxicity_dataset:text:toxic:en" \
          "AIM-Harvard/reject_prompts:text:label:train" \
          "jackhhao/jailbreak-classification:prompt:type:test")

for dataset in "${DATASETS[@]}"; do
    IFS=":" read -r dataset_name text_field label_field dataset_split <<< "$dataset"
    save_dir="${BASE_SAVE_DIR}/${MODEL_NAME}/width_${WIDTH}/${dataset_name}_${dataset_split}"
    input_dir="${BASE_INPUT_DIR}/${MODEL_NAME}/width_${WIDTH}"

    run_extraction "${MODEL_NAME}" "${LAYERS}" "${WIDTH}" "${dataset_name}" "${text_field}" "${label_field}" "${dataset_split}"
    run_classification "${input_dir}" "${MODEL_NAME}" "${dataset_name}" "${LAYERS}" "${WIDTH}" "${dataset_split}"
done

# Gemma 2 9B (width 131k and 1m)
for WIDTH in "131k" "1m"; do
    MODEL_NAME="google/gemma-2-9b"
    LAYERS="9,20,31"
    DATASETS=("AIM-Harvard/sorrybench:turns:category:train" \
              "Anthropic/election_questions:question:label:test" \
              "textdetox/multilingual_toxicity_dataset:text:toxic:en" \
              "AIM-Harvard/reject_prompts:text:label:train" \
              "jackhhao/jailbreak-classification:prompt:type:test")

    for dataset in "${DATASETS[@]}"; do
        IFS=":" read -r dataset_name text_field label_field dataset_split <<< "$dataset"
        save_dir="${BASE_SAVE_DIR}/${MODEL_NAME}/width_${WIDTH}/${dataset_name}_${dataset_split}"
        input_dir="${BASE_INPUT_DIR}/${MODEL_NAME}/width_${WIDTH}"

        run_extraction "${MODEL_NAME}" "${LAYERS}" "${WIDTH}" "${dataset_name}" "${text_field}" "${label_field}" "${dataset_split}"
        run_classification "${input_dir}" "${MODEL_NAME}" "${dataset_name}" "${LAYERS}" "${WIDTH}" "${dataset_split}"
    done
done

# Gemma 2 9B IT (width 131k and 1m)
for WIDTH in "131k" "1m"; do
    MODEL_NAME="google/gemma-2-9b-it"
    LAYERS="9,20,31"
    DATASETS=("AIM-Harvard/sorrybench:turns:category:train" \
              "Anthropic/election_questions:question:label:test" \
              "textdetox/multilingual_toxicity_dataset:text:toxic:en" \
              "AIM-Harvard/reject_prompts:text:label:train" \
              "jackhhao/jailbreak-classification:prompt:type:test")

    for dataset in "${DATASETS[@]}"; do
        IFS=":" read -r dataset_name text_field label_field dataset_split <<< "$dataset"
        save_dir="${BASE_SAVE_DIR}/${MODEL_NAME}/width_${WIDTH}/${dataset_name}_${dataset_split}"
        input_dir="${BASE_INPUT_DIR}/${MODEL_NAME}/width_${WIDTH}"

        run_extraction "${MODEL_NAME}" "${LAYERS}" "${WIDTH}" "${dataset_name}" "${text_field}" "${label_field}" "${dataset_split}"
        run_classification "${input_dir}" "${MODEL_NAME}" "${dataset_name}" "${LAYERS}" "${WIDTH}" "${dataset_split}"
    done
done

echo "All extraction and classification processes completed at: $(date)"
