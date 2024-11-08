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

echo "Starting extractions at: $(date)"

# Process each dataset for Gemma 1 2B
echo "Processing Gemma 1 2B configurations..."
run_extraction "google/gemma-2b" "17" "16k" "sorry-bench/sorry-bench-202406" "turns" "category" "train"
run_extraction "google/gemma-2b" "6,12,17" "16k" "Anthropic/election_questions" "question" "label" "test"
run_extraction "google/gemma-2b" "6,12,17" "16k" "textdetox/multilingual_toxicity_dataset" "text" "toxic" "en"
run_extraction "google/gemma-2b" "6,12,17" "16k" "AIM-Harvard/reject_prompts" "text" "label" "train"
run_extraction "google/gemma-2b" "6,12,17" "16k" "jackhhao/jailbreak-classification" "prompt" "type" "test"

# Process each dataset for Gemma 2 2B (all widths)
for width in "16k" "65k"; do # 1m too much ram
    echo "Processing Gemma 2 2B configurations (${width} width)..."
    run_extraction "google/gemma-2-2b" "5,12,19" "${width}" "sorry-bench/sorry-bench-202406" "turns" "category" "train"
    run_extraction "google/gemma-2-2b" "5,12,19" "${width}" "Anthropic/election_questions" "question" "label" "test"
    run_extraction "google/gemma-2-2b" "5,12,19" "${width}" "textdetox/multilingual_toxicity_dataset" "text" "toxic" "en"
    run_extraction "google/gemma-2-2b" "5,12,19" "${width}" "AIM-Harvard/reject_prompts" "text" "label" "train"
    run_extraction "google/gemma-2-2b" "5,12,19" "${width}" "jackhhao/jailbreak-classification" "prompt" "type" "test"
done

# Process each dataset for Gemma 2 9B (all widths)
for width in "16k" "131k"; do
    echo "Processing Gemma 2 9B configurations (${width} width)..."
    run_extraction "google/gemma-2-9b" "9,20,31" "${width}" "sorry-bench/sorry-bench-202406" "turns" "category" "train"
    run_extraction "google/gemma-2-9b" "9,20,31" "${width}" "Anthropic/election_questions" "question" "label" "test"
    run_extraction "google/gemma-2-9b" "9,20,31" "${width}" "textdetox/multilingual_toxicity_dataset" "text" "toxic" "en"
    run_extraction "google/gemma-2-9b" "9,20,31" "${width}" "AIM-Harvard/reject_prompts" "text" "label" "train"
    run_extraction "google/gemma-2-9b" "9,20,31" "${width}" "jackhhao/jailbreak-classification" "prompt" "type" "test"
done

# Process each dataset for Gemma 2 9B IT (all widths)
for width in "16k" "131k"; do
    echo "Processing Gemma 2 9B IT configurations (${width} width)..."
    run_extraction "google/gemma-2-9b-it" "9,20,31" "${width}" "sorry-bench/sorry-bench-202406" "turns" "category" "train"
    run_extraction "google/gemma-2-9b-it" "9,20,31" "${width}" "Anthropic/election_questions" "question" "label" "test"
    run_extraction "google/gemma-2-9b-it" "9,20,31" "${width}" "textdetox/multilingual_toxicity_dataset" "text" "toxic" "en"
    run_extraction "google/gemma-2-9b-it" "9,20,31" "${width}" "AIM-Harvard/reject_prompts" "text" "label" "train"
    run_extraction "google/gemma-2-9b-it" "9,20,31" "${width}" "jackhhao/jailbreak-classification" "prompt" "type" "test"
done

echo "All extractions completed at: $(date)"
