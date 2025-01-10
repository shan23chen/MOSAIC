#!/bin/bash
# Shell script to run senti_multilingual.py with various arguments

PYTHON_EXEC=python3
SCRIPT="translation_jobs/senti_multilingual.py" # translation_jobs/toxi_multilingual.py
TOP_N=0

# Define arrays for the parameters to loop over
PATH_INDICES=(0 1 2 3)
FEATURES_ARRAY=("features" "hidden_states")
WIDTH_ARRAY=(16 131)

# Loop over each combination of PATH_INDEX, FEATURES, and WIDTH
for PATH_INDEX in "${PATH_INDICES[@]}"; do
  for FEATURES in "${FEATURES_ARRAY[@]}"; do
    for WIDTH in "${WIDTH_ARRAY[@]}"; do
      echo "Running with path_index=${PATH_INDEX}, features=${FEATURES}, width=${WIDTH}"
      $PYTHON_EXEC $SCRIPT --path $PATH_INDEX --features $FEATURES --top_n $TOP_N --width $WIDTH
    done
  done
done