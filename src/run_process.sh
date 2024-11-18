#!/bin/sh

# Base input directory
BASE_INPUT_DIR="./outputs"
DASHBOARD_DIR="../dashboard_data"
MODEL_TYPE="llm"
SAE_LOCATION="res"
TOP_N=5
TEST_SIZE=0.2
TREE_DEPTH=5

# Set GPU for processing
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0

# Function to run dataset classification with error handling
run_classification() {
    input_dir=$1
    model_name=$2
    checkpoint=$3
    dataset_name=$4
    layers=$5
    width=$6
    dataset_split=$7

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

# Process each dataset for Gemma 1 2B
MODEL_NAME="google/gemma-2b"
LAYERS="6"
for width in "16k"; do
    for dataset in "sorry-bench/sorry-bench-202406:train" "Anthropic/election_questions:test" \
                   "textdetox/multilingual_toxicity_dataset:en" "AIM-Harvard/reject_prompts:train" \
                   "jackhhao/jailbreak-classification:test"; do
        dataset_name=$(echo $dataset | cut -d':' -f1)
        dataset_split=$(echo $dataset | cut -d':' -f2)
        model_short_name=$(echo ${MODEL_NAME} | cut -d'/' -f2)
        input_dir="${BASE_INPUT_DIR}/${model_short_name}/width_${width}"
        run_classification ${input_dir} ${MODEL_NAME} ${MODEL_NAME} ${dataset_name} ${LAYERS} ${width} ${dataset_split}
    done
done

# # Process each dataset for Gemma 2 2B
# MODEL_NAME="google/gemma-2-2b"
# LAYERS="5,12,19"
# for width in "16k" "65k"; do
#     for dataset in "sorry-bench/sorry-bench-202406:train" "Anthropic/election_questions:test" \
#                    "textdetox/multilingual_toxicity_dataset:en" "AIM-Harvard/reject_prompts:train" \
#                    "jackhhao/jailbreak-classification:test"; do
#         dataset_name=$(echo $dataset | cut -d':' -f1)
#         dataset_split=$(echo $dataset | cut -d':' -f2)
#         input_dir="${BASE_INPUT_DIR}/gemma-2-2b/width_${width}"
#         run_classification ${input_dir} ${MODEL_NAME} ${MODEL_NAME} ${dataset_name} ${LAYERS} ${width} ${dataset_split}
#     done
# done

# # Process each dataset for Gemma 2 9B
# MODEL_NAME="google/gemma-2-9b"
# LAYERS="9,20,31"
# for width in "16k" "131k"; do
#     for dataset in "sorry-bench/sorry-bench-202406:train" "Anthropic/election_questions:test" \
#                    "textdetox/multilingual_toxicity_dataset:en" "AIM-Harvard/reject_prompts:train" \
#                    "jackhhao/jailbreak-classification:test"; do
#         dataset_name=$(echo $dataset | cut -d':' -f1)
#         dataset_split=$(echo $dataset | cut -d':' -f2)
#         input_dir="${BASE_INPUT_DIR}/gemma-2-9b/width_${width}"
#         run_classification ${input_dir} ${MODEL_NAME} ${MODEL_NAME} ${dataset_name} ${LAYERS} ${width} ${dataset_split}
#     done
# done

# # Process each dataset for Gemma 2 9B IT
# MODEL_NAME="google/gemma-2-9b-it"
# LAYERS="9,20,31"
# for width in "16k" "131k"; do
#     for dataset in "sorry-bench/sorry-bench-202406:train" "Anthropic/election_questions:test" \
#                    "textdetox/multilingual_toxicity_dataset:en" "AIM-Harvard/reject_prompts:train" \
#                    "jackhhao/jailbreak-classification:test"; do
#         dataset_name=$(echo $dataset | cut -d':' -f1)
#         dataset_split=$(echo $dataset | cut -d':' -f2)
#         input_dir="${BASE_INPUT_DIR}/gemma-2-9b-it/width_${width}"
#         run_classification ${input_dir} ${MODEL_NAME} ${MODEL_NAME} ${dataset_name} ${LAYERS} ${width} ${dataset_split}
#     done
# done

echo "All classifications completed at: $(date)"
