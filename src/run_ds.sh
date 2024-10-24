export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0

# python run.py --model_name Intel/llava-gemma-2b \
#     --model_type vlm \
#     --sae_release gemma-2b \
#     --layer 12 \
#     --checkpoint Intel/llava-gemma-2b \
#     --save_dir ./output_vlm_both \
#     --dataset_name renumics/cifar100-enriched \
#     --dataset_split test \
#     --text_field fine_label_str \
#     --image_field full_image \
#     --label_field fine_label_str \
#     --act_only False \
#     --max_batches 3

python run.py --model_name google/gemma-2b-it \
    --model_type llm \
    --sae_release gemma-2b \
    --layer 12 \
    --checkpoint google/gemma-2b-it \
    --save_dir ./output_llm_both \
    --dataset_name shanchen/OncQA \
    --dataset_split train \
    --text_field question \
    --image_field NA \
    --label_field q1 \
    --act_only False \
    --max_batches 3 