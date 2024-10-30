export CUDA_VISIBLE_DEVICES=1  # Use only GPU 0

python run.py --model_name Intel/llava-gemma-2b \
    --model_type vlm \
    --layer 6,10,17 \
    --save_dir ./output_vlm_both_new \
    --dataset_name renumics/cifar100-enriched \
    --dataset_split test \
    --text_field fine_label_str \
    --image_field full_image \
    --label_field fine_label_str \
    --act_only False \
    --sae_location res \
    --width 16k 
    
# python run.py --model_name google/gemma-2b-it \
#     --model_type llm \
#     --sae_release gemma-2b \
#     --layer 10,12 \
#     --checkpoint google/gemma-2b-it \
#     --save_dir ./output_llm_both/ \
#     --dataset_name Anthropic/election_questions \
#     --dataset_split test \
#     --text_field question \
#     --batch_size 16 \
#     --image_field NA \
#     --label_field label \
#     --act_only False 
#     # --max_batches 3 

# python run.py --model_name google/gemma-2b-it \
#     --model_type llm \
#     --sae_release gemma-2b \
#     --layer 10,12 \
#     --checkpoint google/gemma-2b-it \
#     --save_dir ./output_llm_both/ \
#     --dataset_name textdetox/multilingual_toxicity_dataset \
#     --dataset_split en \
#     --text_field text \
#     --batch_size 16 \
#     --image_field NA \
#     --label_field toxic \
#     --act_only False 
#     # --max_batches 3 

# python run.py --model_name google/gemma-2b-it \
#     --model_type llm \
#     --sae_release gemma-2b \
#     --layer 10,12 \
#     --checkpoint google/gemma-2b-it \
#     --save_dir ./output_llm_both/ \
#     --dataset_name textdetox/multilingual_toxicity_dataset \
#     --dataset_split zh \
#     --text_field text \
#     --batch_size 16 \
#     --image_field NA \
#     --label_field toxic \
#     --act_only False 
#     # --max_batches 3 

# python run.py --model_name google/gemma-2b-it \
#     --model_type llm \
#     --sae_release gemma-2b \
#     --layer 10,12 \
#     --checkpoint google/gemma-2b-it \
#     --save_dir ./output_llm_both/ \
#     --dataset_name AIM-Harvard/reject_prompts \
#     --dataset_split train \
#     --text_field text \
#     --batch_size 16 \
#     --image_field NA \
#     --label_field label \
#     --act_only False 
#     # --max_batches 3 

# python run.py --model_name google/gemma-2b-it \
#     --model_type llm \
#     --sae_release gemma-2b \
#     --layer 10,12 \
#     --checkpoint google/gemma-2b-it \
#     --save_dir ./output_llm_both/ \
#     --dataset_name jackhhao/jailbreak-classification \
#     --dataset_split train \
#     --text_field prompt \
#     --batch_size 4 \
#     --image_field NA \
#     --label_field type \
#     --act_only False 
    # --max_batches 3 

# python process_npz_files.py \
#     --input-dir ./output_llm_both \
#     --model-name google/gemma-2b-it \
#     --model-type llm \
#     --layer 12 \
#     --sae-release gemma-2b \
#     --top-n 5 \
#     --output-dir processed_features_llm \
#     --test-size 0.2 \
#     --tree-depth 5 \
#     --save-plots