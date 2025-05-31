#! /usr/bin/env bash
GPU=0
METHOD_NAME=UniQuanF
bcq=True

# Your folder path
export PYTHONPATH=$(pwd)
model_name_or_path='meta-llama/Meta-Llama-3-8B'

# Settings
dataset=c4
seed=0
n_bits_w=4
cache_dir=./cache

FILE_NAME=${n_bits_w}_${seed}

output_dir=outputs/${METHOD_NAME}/${model_name_or_path}/${FILE_NAME}

# Evaluation
# Select tasks in ['mmlu', 'csr', 'ppl', 'gsm8k']
EVAL_OUTPUT_DIR=./eval_results/${METHOD_NAME}/${model_name_or_path}/${FILE_NAME}
CUDA_VISIBLE_DEVICES=$GPU python src/evaluation.py \
    --tasks mmlu ppl \
    --model $output_dir \
    --output_dir $EVAL_OUTPUT_DIR \
    --cache_dir $cache_dir
