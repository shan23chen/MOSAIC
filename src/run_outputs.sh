
#!/bin/bash

# evaluate_medqa.sh

# Common parameters
# DATASET="jackhhao/jailbreak-classification"
# LABEL_COL="type"
# INPUT_COL="prompt"
DATASET="AIM-Harvard/reject_prompts"
LABEL_COL="label"
INPUT_COL="text"
SPLIT="train"
MODEL="openai/gpt-3.5-turbo"  # Replace with your model name
OUTPUT_DIR="./outputs"

#### API ####

# 1. Evaluation with API model
echo "Running evaluation on Jailbreak Classification dataset..."
python step3_evaluate_outputs.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --input-column "$INPUT_COL" \
    --label-column "$LABEL_COL" \
    --eval-type "classification" \
    --scorer "model_graded_qa" \
    --output-dir "${OUTPUT_DIR}/jailbreak_classification" \
    --debug

echo "Evaluation complete. Results are in ${OUTPUT_DIR}/jailbreak_classification"



# #### VLLM ####

# Common parameters
MODEL=unsloth/mistral-7b-instruct-v0.2-bnb-4bit 

# 2. Multiple Choice Evaluation with vllm model and api Model Graded QA
echo "Running Multiple Choice Evaluation with Model Graded QA..."
python step3_evaluate_outputs.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --input-column "$INPUT_COL" \
    --label-column "$LABEL_COL" \
    --id-column "$ID_COL" \
    --choice-columns "$CHOICES" \
    --eval-type "classification" \
    --scorer "model_graded_qa" \
    --scorer-args "instructions=Please evaluate if the model's answer contains the correct medical knowledge and reasoning, even if expressed differently from the reference answer. Grade as C (correct) or I (incorrect)." \
    --output-dir "${OUTPUT_DIR}" \
    --debug \
    --use-vllm \
    --vllm-model-path unsloth/mistral-7b-instruct-v0.2-bnb-4bit \
    --vllm-quantization bitsandbytes \
    --vllm-load-format bitsandbytes \
    --max-connections 32 



# echo "All evaluations complete. Results are in ${OUTPUT_DIR}