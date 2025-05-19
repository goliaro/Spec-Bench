#!/bin/bash
set -x
set -e

cd "$(dirname "$0")"


eval "$(mamba shell hook --shell bash)"
mamba activate specbench


Vicuna_PATH=meta-llama/Llama-3.1-8B-Instruct
Eagle_PATH=yuhuili/EAGLE-LLaMA3.1-Instruct-8B
Eagle3_PATH=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B

MODEL_NAME=Llama-3.1-8B-Instruct
TEMP=0.0

bench_NAME="swebench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]
max_new_tokens=2048
    
CUDA_VISIBLE_DEVICES=0 python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME  --temperature $TEMP --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/swebench_baseline.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m evaluation.inference_eagle --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle-${torch_dtype} --bench-name $bench_NAME  --temperature $TEMP --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/swebench_eagle.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -m evaluation.inference_eagle2 --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle2-${torch_dtype} --bench-name $bench_NAME  --temperature $TEMP --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/swebench_eagle2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -m evaluation.inference_eagle3 --ea-model-path $Eagle3_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle3-${torch_dtype} --bench-name $bench_NAME  --temperature $TEMP --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/swebench_eagle3.txt 2>&1 &

CUDA_VISIBLE_DEVICES=4 USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME  --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/swebench_lookahead.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME  --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/swebench_pld.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 python -m evaluation.inference_recycling --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-recycling --bench-name $bench_NAME  --temperature $TEMP --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/swebench_recycling.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 python -m evaluation.inference_suffix --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-suffix-${torch_dtype} --bench-name $bench_NAME  --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/swebench_suffix.txt 2>&1 &

wait

