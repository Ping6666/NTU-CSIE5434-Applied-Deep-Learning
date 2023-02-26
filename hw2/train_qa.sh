#!/bin/bash

# "${1}": path to the context file.
# "${2}": path to the train file.
# "${3}": path to the valid file.

python ./preprocess/json_adjustment.py 'qa' "${1}" "${2}" ./tmp/qa_train_adjustment.json
python ./preprocess/json_adjustment.py 'qa' "${1}" "${3}" ./tmp/qa_valid_adjustment.json

# num_train_epochs: 1~3

## pretrained ##
# --model_name_or_path "{model_name_on_huggingface_}"

## from scratch ##
# --model_type "{name_in_MODEL_MAPPING_NAMES}"
# --tokenizer_name "{model_name_on_huggingface_}"

## run_qa ##
python ./question_answering/run_qa.py \
    --model_name_or_path "hfl/chinese-lert-base" \
    --output_dir ./best/qa/ \
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
    --train_file ./tmp/qa_train_adjustment.json \
    --do_eval \
    --validation_file ./tmp/qa_valid_adjustment.json

## run_qa_no_trainer ##
# python ./question_answering/run_qa_no_trainer.py \
#     --model_name_or_path "hfl/chinese-lert-base" \
#     --output_dir ./best/qa/ \
#     --max_seq_length 512 \
#     --pad_to_max_length \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 2 \
#     --num_train_epochs 3 \
#     --learning_rate 3e-5 \
#     --train_file ./tmp/qa_train_adjustment.json \
#     --validation_file ./tmp/qa_valid_adjustment.json
