#!/bin/bash
#SBATCH --job-name=eval_array
#SBATCH --output=logs/eval_array_%A_%a.out
#SBATCH --error=logs/eval_array_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --array=0-9

# Set environment variables (copied from eval.sh)
Vicuna_PATH=lmsys/vicuna-13b-v1.3
Eagle_PATH=yuhuili/EAGLE-Vicuna-13B-v1.3
Eagle3_PATH=yuhuili/EAGLE3-Vicuna1.3-13B
Medusa_PATH=FasterDecoding/medusa-vicuna-13b-v1.3
# Hydra_PATH=/your_own_path/hydra-vicuna-7b-v1.3
# Drafter_PATH=/your_own_path/vicuna-68m
# Space_PATH=/your_own_path/vicuna-v1.3-7b-space
datastore_PATH=./model/rest/datastore/datastore_chat_large.idx
MODEL_NAME=vicuna-13b-v1.3
TEMP=0.0
GPU_DEVICES=0

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

# Load modules or conda env if needed
# module load ...
# source activate ...

mkdir -p logs

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_medusa --model-path $Medusa_PATH --base-model $Vicuna_PATH --model-id ${MODEL_NAME}-medusa-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle2 --ea-model-path $Eagle_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle2-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_eagle3 --ea-model-path $Eagle3_PATH --base-model-path $Vicuna_PATH --model-id ${MODEL_NAME}-eagle3-${torch_dtype} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
elif [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} USE_LADE=1 python -m evaluation.inference_lookahead --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-lade-level-5-win-7-guess-7-${torch_dtype} --level 5 --window 7 --guess 7 --bench-name $bench_NAME --dtype $torch_dtype
elif [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype
elif [ "$SLURM_ARRAY_TASK_ID" -eq 7 ]; then
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} RAYON_NUM_THREADS=6 python -m evaluation.inference_rest --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-rest-${torch_dtype} --datastore-path $datastore_PATH --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
elif [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-recycling --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
elif [ "$SLURM_ARRAY_TASK_ID" -eq 9 ]; then
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_samd --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-samd --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle2 --attn_implementation sdpa --tree_model_path $Eagle_PATH
else
  echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
  exit 1
fi 