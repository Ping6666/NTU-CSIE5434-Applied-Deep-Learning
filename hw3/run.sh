#!/bin/bash

# ${1}: path to the input file
# ${2}: path to the output file

python ./preprocess/jsonl_adjustment.py ${1} ./tmp/test.json

# CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=2 \

## inference ##
python ./summarization/inference.py \
    --model_name_or_path "google/mt5-small" \
    --cache_dir "./best/cache/" \
    --pt_path "./best/best.pt" \
    --test_file ./tmp/test.json \
    --inference_file ${2}
