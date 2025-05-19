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

bench_NAME="cortex"
partitions=(CATEGORIZATION FEATURE_EXTRACTION QUESTION_SUGGESTION SQL_FANOUT1 SQL_FANOUT2 SQL_FANOUT3 SQL_COMBINE)
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]
max_new_tokens=2048


for partition in "${partitions[@]}"; do
    
    echo "Running evaluation for partition: $partition"
    
    CUDA_VISIBLE_DEVICES=0 python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --partition-name "$partition" --temperature $TEMP --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/logs_baseline_"$partition".txt 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python -m evaluation.inference_eagle --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle-${torch_dtype} --bench-name $bench_NAME --partition-name "$partition" --temperature $TEMP --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/logs_eagle_"$partition".txt 2>&1 &

    CUDA_VISIBLE_DEVICES=2 python -m evaluation.inference_eagle2 --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle2-${torch_dtype} --bench-name $bench_NAME --partition-name "$partition" --temperature $TEMP --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/logs_eagle2_"$partition".txt 2>&1 &
    CUDA_VISIBLE_DEVICES=3 python -m evaluation.inference_eagle3 --ea-model-path $Eagle3_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle3-${torch_dtype} --bench-name $bench_NAME --partition-name "$partition" --temperature $TEMP --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/logs_eagle3_"$partition".txt 2>&1 &

    # CUDA_VISIBLE_DEVICES=4 USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME --partition-name "$partition" --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/logs_lookahead_"$partition".txt 2>&1 &
    CUDA_VISIBLE_DEVICES=5 python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --partition-name "$partition" --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/logs_pld_"$partition".txt 2>&1 &
    CUDA_VISIBLE_DEVICES=6 python -m evaluation.inference_recycling --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-recycling --bench-name $bench_NAME --partition-name "$partition" --temperature $TEMP --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/logs_recycling_"$partition".txt 2>&1 &
    #CUDA_VISIBLE_DEVICES=7 python -m evaluation.inference_samd --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-samd --bench-name $bench_NAME --partition-name "$partition" --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle2 --attn_implementation sdpa --tree_model_path $Eagle_PATH --max-new-tokens $max_new_tokens > logs/logs_samd_"$partition".txt 2>&1 &
    CUDA_VISIBLE_DEVICES=7 python -m evaluation.inference_suffix --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-suffix-${torch_dtype} --bench-name $bench_NAME --partition-name "$partition" --dtype $torch_dtype --max-new-tokens $max_new_tokens > logs/logs_suffix_"$partition".txt 2>&1 &

    wait
done


