"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import os
from fastchat.utils import str_to_torch_dtype
from model.recycling.recycling import TokenRecycling, Outputs
from evaluation.eval import run_eval, reorg_answer_file


def recycling_forward(inputs, recycler, tokenizer, max_new_tokens, temperature=0.0, do_sample=False):
    outputs = recycler.generate(prompt=inputs.input_ids, max_new_tokens=max_new_tokens, hot_start=False, silent=True, stop_on_eos=True)
    accept_length_list = [len(seq) for seq in outputs.accepted_sequences]
    num_accepted_tokens = sum(accept_length_list)
    num_steps = outputs.total_steps
    return outputs.output_ids, num_accepted_tokens, num_steps, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--partition-name",
        type=str,
        default="",
        help="The partition of the dataset to use.",
    )

    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"

    question_folder = f"data/{args.bench_name}"
    question_filename = "question.jsonl"
    if args.bench_name == "cortex" or args.bench_name == "swebench":
        partition_suffix = f"_{args.partition_name}" if len(args.partition_name) > 0 else ""
        question_filename = f"eval{partition_suffix}.jsonl"
    question_file = os.path.join(question_folder, question_filename)
    if args.answer_file:
        answer_file = args.answer_file
    else:
        partition_prefix = f"{args.partition_name + '_' if len(args.partition_name) > 0 else ''}"
        answer_file = f"data/{args.bench_name}/model_answer/{partition_prefix}{args.model_id}.jsonl"
    print("Loading question file:", question_file)
    print(f"Output to {answer_file}")

    recycler = TokenRecycling.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval(
        model=recycler,
        tokenizer=recycler.tokenizer,
        forward_func=recycling_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        do_sample=do_sample,
    )

    reorg_answer_file(answer_file)