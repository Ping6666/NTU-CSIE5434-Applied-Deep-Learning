#!/bin/bash

# make dir
mkdir -p ./best/

# download pretrained model to cache
python ./preprocess/pre_download.py \
    --model_name_or_path "google/mt5-small" \
    --cache_dir "./best/cache/"

# download my fine-tuned model
wget -O ./best/best.pt $dropbox_link
