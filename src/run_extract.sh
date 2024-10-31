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

###################################### Anthropic ##############################################
python run.py --model_name google/gemma-2-2b \
    --model_type llm \
    --sae_location res \
    --layer 5,12,19 \
    --save_dir ./output_llm_both/ \
    --dataset_name Anthropic/election_questions \
    --dataset_split test \
    --text_field question \
    --batch_size 16 \
    --image_field NA \
    --label_field label \
    --act_only False \
    --width 16k
    # --max_batches 3 

    
# python run.py --model_name google/gemma-2-9b \
#     --model_type llm \
#     --sae_location res \
#     --layer 9,20,31 \
#     --save_dir ./output_llm_both/ \
#     --dataset_name Anthropic/election_questions \
#     --dataset_split test \
#     --text_field question \
#     --batch_size 2 \
#     --image_field NA \
#     --label_field label \
#     --act_only False \
#     --width 16k
 
#     # --max_batches 3 
# python run.py --model_name google/gemma-2-9b-it \
#     --model_type llm \
#     --sae_location res \
#     --layer 9,20,31 \
#     --save_dir ./output_llm_both/ \
#     --dataset_name Anthropic/election_questions \
#     --dataset_split test \
#     --text_field question \
#     --batch_size 2 \
#     --image_field NA \
#     --label_field label \
#     --act_only False \
#     --width 16k

###################################### TextDetox EN ##############################################

python run.py --model_name google/gemma-2-2b \
    --model_type llm \
    --sae_location res \
    --layer 5,12,19 \
    --save_dir ./output_llm_both/ \
    --dataset_name textdetox/multilingual_toxicity_dataset \
    --dataset_split en \
    --text_field text \
    --batch_size 16 \
    --image_field NA \
    --label_field toxic \
    --act_only False \
    --width 16k
    # --max_batches 3 
    
# python run.py --model_name google/gemma-2-9b \
#     --model_type llm \
#     --sae_location res \
#     --layer 9,20,31 \
#     --save_dir ./output_llm_both/ \
#     --dataset_name textdetox/multilingual_toxicity_dataset \
#     --dataset_split en \
#     --text_field text \
#     --batch_size 2 \
#     --image_field NA \
#     --label_field toxic \
#     --act_only False \
#     --width 16k
#     # --max_batches 3 
 
#     # --max_batches 3 
# python run.py --model_name google/gemma-2-9b-it \
#     --model_type llm \
#     --sae_location res \
#     --layer 9,20,31 \
#     --save_dir ./output_llm_both/ \
#     --dataset_name textdetox/multilingual_toxicity_dataset \
#     --dataset_split en \
#     --text_field text \
#     --batch_size 2 \
#     --image_field NA \
#     --label_field toxic \
#     --act_only False \
#     --width 16k
#     # --max_batches 3 

###################################### TextDetox ZH ##############################################
python run.py --model_name google/gemma-2-2b \
    --model_type llm \
    --sae_location res \
    --layer 5,12,19 \
    --save_dir ./output_llm_both/ \
    --dataset_name textdetox/multilingual_toxicity_dataset \
    --dataset_split zh \
    --text_field text \
    --batch_size 16 \
    --image_field NA \
    --label_field toxic \
    --act_only False \
    --width 16k
    # --max_batches 3

# python run.py --model_name google/gemma-2-9b \
#     --model_type llm \
#     --sae_location res \
#     --layer 9,20,31 \
#     --save_dir ./output_llm_both/ \
#     --dataset_name textdetox/multilingual_toxicity_dataset \
#     --dataset_split zh \
#     --text_field text \
#     --batch_size 2 \
#     --image_field NA \
#     --label_field toxic \
#     --act_only False \
#     --width 16k
#     # --max_batches 3

# python run.py --model_name google/gemma-2-9b-it \
#     --model_type llm \
#     --sae_location res \
#     --layer 9,20,31 \
#     --save_dir ./output_llm_both/ \
#     --dataset_name textdetox/multilingual_toxicity_dataset \
#     --dataset_split zh \
#     --text_field text \
#     --batch_size 2 \
#     --image_field NA \
#     --label_field toxic \
#     --act_only False \
#     --width 16k
#     # --max_batches 3

###################################### AIM-Harvard ##############################################
python run.py --model_name google/gemma-2-2b \
    --model_type llm \
    --sae_location res \
    --layer 5,12,19 \
    --save_dir ./output_llm_both/ \
    --dataset_name AIM-Harvard/reject_prompts \
    --dataset_split train \
    --text_field text \
    --batch_size 16 \
    --image_field NA \
    --label_field label \
    --act_only False \
    --width 16k
    # --max_batches 3

# python run.py --model_name google/gemma-2-9b \
#     --model_type llm \
#     --sae_location res \
#     --layer 9,20,31 \
#     --save_dir ./output_llm_both/ \
#     --dataset_name AIM-Harvard/reject_prompts \
#     --dataset_split train \
#     --text_field text \
#     --batch_size 2 \
#     --image_field NA \
#     --label_field label \
#     --act_only False \
#     --width 16k
#     # --max_batches 3

# python run.py --model_name google/gemma-2-9b-it \
#     --model_type llm \
#     --sae_location res \
#     --layer 9,20,31 \
#     --save_dir ./output_llm_both/ \
#     --dataset_name AIM-Harvard/reject_prompts \
#     --dataset_split train \
#     --text_field text \
#     --batch_size 2 \
#     --image_field NA \
#     --label_field label \
#     --act_only False \
#     --width 16k
#     # --max_batches 3

###################################### Jailbreak ##############################################
# python run.py --model_name google/gemma-2-2b \
#     --model_type llm \
#     --sae_location res \
#     --layer 5,12,19 \
#     --save_dir ./output_llm_both/ \
#     --dataset_name jackhhao/jailbreak-classification \
#     --dataset_split train \
#     --text_field prompt \
#     --batch_size 16 \
#     --image_field NA \
#     --label_field type \
#     --act_only False \
#     --width 16k
#     # --max_batches 3

# python run.py --model_name google/gemma-2-9b \
#     --model_type llm \
#     --sae_location res \
#     --layer 9,20,31 \
#     --save_dir ./output_llm_both/ \
#     --dataset_name jackhhao/jailbreak-classification \
#     --dataset_split train \
#     --text_field prompt \
#     --batch_size 2 \
#     --image_field NA \
#     --label_field type \
#     --act_only False \
#     --width 16k
#     # --max_batches 3

# python run.py --model_name google/gemma-2-9b-it \
#     --model_type llm \
#     --sae_location res \
#     --layer 9,20,31 \
#     --save_dir ./output_llm_both/ \
#     --dataset_name jackhhao/jailbreak-classification \
#     --dataset_split train \
#     --text_field prompt \
#     --batch_size 2 \
#     --image_field NA \
#     --label_field type \
#     --act_only False \
#     --width 16k
#     # --max_batches 3

