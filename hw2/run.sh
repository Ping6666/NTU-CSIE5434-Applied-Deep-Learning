#!/bin/bash

# "${1}": path to the context file.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.

python ./preprocess/json_adjustment.py 'mc' "${1}" "${2}" ./tmp/mc_test_adjustment.json
# --cache_dir ./best/cache/ \

## run_swag ##
python ./multiple_choice/run_swag.py \
    --model_name_or_path ./best/mc/ \
    --output_dir ./best/mc/ \
    --max_seq_length 512 \
    --pad_to_max_length \
    --per_device_eval_batch_size 1 \
    --do_predict \
    --test_file ./tmp/mc_test_adjustment.json \
    --output_file ./mc_pred.json

## run_qa ##
python ./question_answering/run_qa.py \
    --model_name_or_path ./best/qa/ \
    --output_dir ./best/qa/ \
    --max_seq_length 512 \
    --pad_to_max_length \
    --per_device_eval_batch_size 1 \
    --do_predict \
    --test_file ./mc_pred.json \
    --pred_file "${3}"
