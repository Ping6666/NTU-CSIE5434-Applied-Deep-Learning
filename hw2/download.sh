#!/bin/bash

# python ./preprocess/pre_download.py \
#     --model_name_or_path "hfl/chinese-lert-base" \
#     --cache_dir ./best/cache/

wget -O ./src/$filename.zip $dropbox_link
unzip ./src/$filename.zip

## files in .zip ##
# $filename.zip
# |-- mc
# |   |-- config.json
# |   |-- pytorch_model.bin
# |   |-- README.md
# |   |-- special_tokens_map.json
# |   |-- tokenizer_config.json
# |   |-- tokenizer.json
# |   |-- training_args.bin
# |   `-- vocab.txt
# `-- qa
#     |-- config.json
#     |-- predict_nbest_predictions.json
#     |-- predict_predictions.json
#     |-- pytorch_model.bin
#     |-- README.md
#     |-- special_tokens_map.json
#     |-- tokenizer_config.json
#     |-- tokenizer.json
#     |-- training_args.bin
#     `-- vocab.txt
