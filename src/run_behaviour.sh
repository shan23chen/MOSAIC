#!/bin/bash

# if want to use vllm start the server before e.g
# vllm serve google/gemma-2-9b --enforce-eager --port=8080 --gpu-memory-utilization=0.95 --dtype=auto --disable-sliding-window --trust-remote-code --tensor-parallel-size=2 --chat-template=gemma_chat.jinja# you also need to change the model name from hf/model -> openai/model so inspect can find the model
# if not then you do not need the model base url 

# Configuration for model graded fact evaluation
model_base_url="http://localhost:8080/v1"

# Configuration
SCORER_MODEL="openai/gpt-4o-mini"

mkdir -p logs
chmod 755 logs

# # Default inspect tasks
# default_inspect_tasks=(
#     "inspect_evals/pubmedqa"
#     "inspect_evals/piqa"
#     "inspect_evals/boolq"
# )

# Function to run inspect default evaluations
run_default_inspect_tasks() {
    local model=$1
    local base_url=$2

    echo "Running default inspect tasks..."
    for task in "${default_inspect_tasks[@]}"; do
        echo "Evaluating task: $task with model: $model"
        inspect eval "$task" \
            --model "$model" \
            --model-base-url "$base_url" 

        if [ $? -ne 0 ]; then
            echo "Error evaluating task: $task with model: $model"
            exit 1
        fi
    done
}

# Function to run custom evaluations
run_model_graded_fact_evaluation() {
    local model=$1
    local dataset=$2
    local context_col=$3
    local answer_col=$4
    local model_url=$5

    echo "Evaluating dataset: $dataset with model: $model using context: $context_col and answers: $answer_col"

    python step3_outputs.py \
        --model "$model" \
        --model-base-url "$model_url" \
        --dataset "$dataset" \
        --split "dev" \
        --eval-type "open_ended" \
        --input-column "question" \
        --context-column "$context_col" \
        --label-column "$answer_col" \
        --scorer "model_graded_fact" \
        --scorer-args "model=$SCORER_MODEL" \
        --system-prompt "You are a helpful AI assistant respond to the following questions based on the context provided." \
        --max-connections 32 

    if [ $? -ne 0 ]; then
        echo "Error evaluating dataset: $dataset with model: $model using context: $context_col and answers: $answer_col"
        exit 1
    fi
}

# Start evaluation
echo "Starting Model Graded Fact evaluation..."

# Define the model
models=("openai/google/gemma-2-9b")

# Define dataset-context-answer mappings
dataset_mappings=(
    "pminervini/NQ-Swap:org_context:org_answer"
    "pminervini/NQ-Swap:sub_context:sub_answer"
    "pminervini/NQ-Swap:none:org_answer"
    # "your/new-dataset:new_context:new_answer"
    # google-research-datasets/nq_open
)

# # # Run default inspect tasks
# for model in "${models[@]}"; do
#     run_default_inspect_tasks "$model" "$model_base_url"
# done

# Iterate over models and dataset mappings for custom tasks
for model in "${models[@]}"; do
    for mapping in "${dataset_mappings[@]}"; do
        IFS=":" read -r dataset context_col answer_col <<< "$mapping"
        run_model_graded_fact_evaluation "$model" "$dataset" "$context_col" "$answer_col" "$model_base_url"
    done
done

# View logs
echo "Evaluation complete- View the logs at $INSPECT_LOG_DIR with-- inspect view"
