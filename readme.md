SAE as features -> explainable classifier

finished VLLM/LLM loading and extract all hidden given layer

then map to SAE and get features -> see playground.ipynb for step by step and process_npz_files for all

Can store both activations and SAE, but do not save all SAE because they are expensive to store

TODO next week -> connect them to classifiers, look into classifier visualizations
