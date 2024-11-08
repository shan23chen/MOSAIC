python main.py --vlm_model Intel/llava-gemma-2b \
    --sae_layer 6 \
    --classifier ../test_model.joblib \
    --neuronpedia_cache ../src/explanation_cache/gemma-2-2b_layer5_width_16k_gemma-scope-2b-pt-res-canonical.json \
    --debug
