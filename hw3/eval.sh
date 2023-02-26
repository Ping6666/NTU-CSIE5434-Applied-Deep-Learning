#!/bin/bash

# ${1}: path to the reference file
# ${2}: path to the submission file

# CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=2 \

## eval ##
python ./summarization/eval.py -r ${1} -s ${2}
