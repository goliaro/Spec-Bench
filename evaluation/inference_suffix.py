import argparse
import os

from evaluation.eval import run_eval, reorg_answer_file

from fastchat.utils import str_to_torch_dtype

from transformers import StoppingCriteriaList, MaxLengthCriteria
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.suffix.suffix import greedy_search_suffix, SuffixCache, SuffixTree, Candidate


def suffix_forward(inputs, model, tokenizer, max_new_tokens):
    input_ids = inputs.input_ids
    output_ids, idx, accept_length_list = model.greedy_search_suffix(
              inputs.input_ids,
              attention_mask=inputs.attention_mask,
              stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=len(inputs.input_ids[0]) + max_new_tokens)]),
              draft_matching_window_size=3,
              draft_num_candidate_tokens=10,
              use_cache=True,
              pad_token_id=tokenizer.pad_token_id,
              eos_token_id=tokenizer.eos_token_id,
              return_dict_in_generate=False)
    input_len = len(input_ids[0])
    new_token = len(output_ids[0][input_len:])
    if tokenizer.eos_token_id in output_ids[0, input_len:].tolist():
        for i, id in enumerate(output_ids[0, input_len:]):
            if id == tokenizer.eos_token_id:
                eos_token_ids_index = i
        invalid_len = len(output_ids[0, input_len:]) - eos_token_ids_index - 1
        if invalid_len > 0:
            accept_length_list[-1] -= invalid_len
            new_token -= invalid_len
    return output_ids, new_token, idx+1, accept_length_list


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
    question_folder = f"data/{args.bench_name}"
    question_filename = "question.jsonl"
    training_file=""
    if args.bench_name == "cortex" or args.bench_name == "swebench":
        partition_suffix = f"_{args.partition_name}" if len(args.partition_name) > 0 else ""
        question_filename = f"eval{partition_suffix}.jsonl"
        train_filename = f"train{partition_suffix}.jsonl"
        training_file = os.path.join(question_folder, train_filename)
    question_file = os.path.join(question_folder, question_filename)
    if not os.path.exists(training_file):
        training_file = None
    if args.answer_file:
        answer_file = args.answer_file
    else:
        partition_prefix = f"{args.partition_name + '_' if len(args.partition_name) > 0 else ''}"
        answer_file = f"data/{args.bench_name}/model_answer/{partition_prefix}{args.model_id}.jsonl"
    print("Loading question file:", question_file)
    print("Loading training file:", training_file)
    print(f"Output to {answer_file}")


    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model._suffix_cache = SuffixCache(64, training_file, tokenizer)
    model.greedy_search_suffix = greedy_search_suffix.__get__(model, type(model))

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=suffix_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
    )

    reorg_answer_file(answer_file)