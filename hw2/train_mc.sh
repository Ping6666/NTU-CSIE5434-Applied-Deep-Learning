#!/bin/bash

# "${1}": path to the context file.
# "${2}": path to the train file.
# "${3}": path to the valid file.

python ./preprocess/json_adjustment.py 'mc' "${1}" "${2}" ./tmp/mc_train_adjustment.json
python ./preprocess/json_adjustment.py 'mc' "${1}" "${3}" ./tmp/mc_valid_adjustment.json

## pretrained ##
# --model_name_or_path "{model_name_on_huggingface_}"

## from scratch ##
# --model_type "{name_in_MODEL_MAPPING_NAMES}"
# --tokenizer_name "{model_name_on_huggingface_}"

## run_swag ##
python ./multiple_choice/run_swag.py \
    --model_name_or_path "hfl/chinese-lert-base" \
    --output_dir ./best/mc/ \
    --overwrite_output \
    --max_seq_length 512 \
    --pad_to_max_length \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --learning_rate 3e-5 \
    --save_strategy=no \
    --evaluation_strategy=steps \
    --do_train \
    --train_file ./tmp/mc_train_adjustment.json \
    --do_eval \
    --validation_file ./tmp/mc_valid_adjustment.json

## run_swag_no_trainer ##
# python ./multiple_choice/run_swag_no_trainer.py \
#     --model_name_or_path "hfl/chinese-lert-base" \
#     --output_dir ./best/mc/ \
#     --max_length 512 \
#     --pad_to_max_length \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 2 \
#     --num_train_epochs 1 \
#     --learning_rate 3e-5 \
#     --train_file ./tmp/mc_train_adjustment.json \
#     --validation_file ./tmp/mc_valid_adjustment.json
