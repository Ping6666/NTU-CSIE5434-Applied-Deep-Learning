#!/bin/bash

python ./preprocess/jsonl_adjustment.py ./data/train.jsonl ./tmp/train.json
python ./preprocess/jsonl_adjustment.py ./data/public.jsonl ./tmp/public.json

# CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=2 \

## my_summarization ##
python ./summarization/train.py \
    --model_name_or_path "google/mt5-small" \
    --cache_dir "./best/cache/" \
    --output_dir "./best" \
    --train_file ./tmp/train.json \
    --validation_file ./tmp/public.json
