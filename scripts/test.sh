#!/bin/bash
set -x
set -e

cd "$(dirname "$0")/.."


eval "$(mamba shell hook --shell bash)"
mamba activate specbench


Vicuna_PATH=meta-llama/Llama-3.1-8B-Instruct
Eagle_PATH=yuhuili/EAGLE-LLaMA3.1-Instruct-8B
Eagle3_PATH=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B

MODEL_NAME=Llama-3.1-8B-Instruct
TEMP=0.0

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=0 python -m evaluation.inference_hybrid \
    --ea-model-path $Eagle3_PATH \
    --base-model-path $Vicuna_PATH \
    --model-id ${MODEL_NAME}-hybrid-${torch_dtype} \
    --bench-name $bench_NAME \
    --question-begin 81 --question-end 82 \
    --temperature $TEMP \
    --dtype $torch_dtype > logs/logs_hybrid.txt 2>&1


