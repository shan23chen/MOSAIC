#!/bin/sh

# Base output directory
BASE_SAVE_DIR="./outputs"

# Common parameters
BATCH_SIZE=2
MODEL_TYPE="llm"
SAE_LOCATION="res"

# Function to run extraction with error handling
run_extraction() {
    model_name=$1
    layers=$2
    width=$3
    dataset_name=$4
    text_field=$5
    label_field=$6
    dataset_split=$7
    
    # Create specific output directory including model and width information
    model_short_name=$(echo ${model_name} | cut -d'/' -f2)
    save_dir="${BASE_SAVE_DIR}/${model_short_name}/width_${width}"
    
    echo "==============================================="
    echo "Starting extraction with configuration:"
    echo "Model: ${model_name}"
    echo "Layers: ${layers}"
    echo "Width: ${width}"
    echo "Dataset: ${dataset_name}"
    echo "Split: ${dataset_split}"
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
        --batch_size ${BATCH_SIZE} \
        --image_field NA \
        --label_field ${label_field} \
        --act_only True \
        --width ${width}
    
    status=$?
    if [ ${status} -eq 0 ]; then
        echo "Successfully completed extraction for ${dataset_name} with ${model_name} (width=${width})"
    else
        echo "Error during extraction for ${dataset_name} with ${model_name} (width=${width})"
        echo "Error code: ${status}"
    fi
}

# Create base output directory
mkdir -p ${BASE_SAVE_DIR}

echo "Starting missing extractions at: $(date)"

# Process missing configurations for Gemma 2 2B (width 1m)
echo "Processing missing Gemma 2 2B configurations (1m width)..."
run_extraction "google/gemma-2-2b" "5,12,19" "1m" "AIM-Harvard/sorrybench" "turns" "category" "train"
run_extraction "google/gemma-2-2b" "5,12,19" "1m" "Anthropic/election_questions" "question" "label" "test"
run_extraction "google/gemma-2-2b" "5,12,19" "1m" "textdetox/multilingual_toxicity_dataset" "text" "toxic" "en"
run_extraction "google/gemma-2-2b" "5,12,19" "1m" "AIM-Harvard/reject_prompts" "text" "label" "train"
run_extraction "google/gemma-2-2b" "5,12,19" "1m" "jackhhao/jailbreak-classification" "prompt" "type" "test"

# Process missing configurations for Gemma 2 9B (width 131k)
echo "Processing missing Gemma 2 9B configurations (131k width)..."
run_extraction "google/gemma-2-9b" "9,20,31" "131k" "AIM-Harvard/sorrybench" "turns" "category" "train"
run_extraction "google/gemma-2-9b" "9,21,31" "131k" "jackhhao/jailbreak-classification" "prompt" "type" "test"

# Process missing configurations for Gemma 2 9B (width 1m)
echo "Processing missing Gemma 2 9B configurations (1m width)..."
run_extraction "google/gemma-2-9b" "9,20,31" "1m" "AIM-Harvard/sorrybench" "turns" "category" "train"
run_extraction "google/gemma-2-9b" "9,20,31" "1m" "Anthropic/election_questions" "question" "label" "test"
run_extraction "google/gemma-2-9b" "9,20,31" "1m" "textdetox/multilingual_toxicity_dataset" "text" "toxic" "en"
run_extraction "google/gemma-2-9b" "9,20,31" "1m" "AIM-Harvard/reject_prompts" "text" "label" "train"
run_extraction "google/gemma-2-9b" "9,20,31" "1m" "jackhhao/jailbreak-classification" "prompt" "type" "test"

# Process missing configurations for Gemma 2 9B IT (width 131k)
echo "Processing missing Gemma 2 9B IT configurations (131k width)..."
run_extraction "google/gemma-2-9b-it" "9,20,31" "131k" "AIM-Harvard/reject_prompts" "text" "label" "train"
run_extraction "google/gemma-2-9b-it" "9,20,31" "131k" "Anthropic/election_questions" "question" "label" "test"
run_extraction "google/gemma-2-9b-it" "9,20,31" "131k" "jackhhao/jailbreak-classification" "prompt" "type" "test"
run_extraction "google/gemma-2-9b-it" "9,20,31" "131k" "textdetox/multilingual_toxicity_dataset" "text" "toxic" "en"

# Process missing configurations for Gemma 2 9B IT (width 1m)
echo "Processing missing Gemma 2 9B IT configurations (1m width)..."
run_extraction "google/gemma-2-9b-it" "9,20,31" "1m" "AIM-Harvard/sorrybench" "turns" "category" "train"
run_extraction "google/gemma-2-9b-it" "9,20,31" "1m" "Anthropic/election_questions" "question" "label" "test"
run_extraction "google/gemma-2-9b-it" "9,20,31" "1m" "textdetox/multilingual_toxicity_dataset" "text" "toxic" "en"
run_extraction "google/gemma-2-9b-it" "9,20,31" "1m" "AIM-Harvard/reject_prompts" "text" "label" "train"
run_extraction "google/gemma-2-9b-it" "9,20,31" "1m" "jackhhao/jailbreak-classification" "prompt" "type" "test"

echo "All missing extractions completed at: $(date)"