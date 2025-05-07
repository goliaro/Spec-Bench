import json
import argparse
from transformers import AutoTokenizer
import numpy as np
import os
import sys


def speed(jsonl_file, jsonl_file_base, tokenizer, task=None, report=True):
    tokenizer=AutoTokenizer.from_pretrained(tokenizer)
    mt_bench_list = ["writing", "roleplay", "reasoning", "math" , "coding", "extraction", "stem", "humanities"]

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            if task=="overall":
                data.append(json_obj)
            elif task == "mt_bench":
                if json_obj["category"] in mt_bench_list:
                    data.append(json_obj)
            else:
                if json_obj["category"] == task:
                    data.append(json_obj)

    speeds=[]
    accept_lengths_list = []
    for datapoint in data:
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        accept_lengths_list.extend(datapoint["choices"][0]['accept_lengths'])
        speeds.append(tokens/times)


    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            if task=="overall":
                data.append(json_obj)
            elif task == "mt_bench":
                if json_obj["category"] in mt_bench_list:
                    data.append(json_obj)
            else:
                if json_obj["category"] == task:
                    data.append(json_obj)

    total_time=0
    total_token=0
    speeds0=[]
    for datapoint in data:
        answer=datapoint["choices"][0]['turns']
        tokens = 0
        for i in answer:
            tokens += (len(tokenizer(i).input_ids) - 1)
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds0.append(tokens / times)
        total_time+=times
        total_token+=tokens

    tokens_per_second = np.array(speeds).mean()
    tokens_per_second_baseline = np.array(speeds0).mean()
    speedup_ratio = np.array(speeds).mean()/np.array(speeds0).mean()

    if report:
        print("="*30, "Task: ", task, "="*30)
        print("#Mean accepted tokens: ", np.mean(accept_lengths_list))
        print('Tokens per second: ', tokens_per_second)
        print('Tokens per second for the baseline: ', tokens_per_second_baseline)
        print("Speedup ratio: ", speedup_ratio)
    return tokens_per_second, tokens_per_second_baseline, speedup_ratio, accept_lengths_list


def get_single_speedup(jsonl_file, jsonl_file_base, tokenizer_path):
    for subtask_name in ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag", "overall"]:
        speed(jsonl_file, jsonl_file_base, tokenizer_path, task=subtask_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        default='./data/spec_bench/model_answer',
        type=str,
        help="The folder containing evaluated Speculative Decoding methods and baseline.",
    )
    parser.add_argument(
        "--tokenizer",
        default='lmsys/vicuna-13b-v1.3',
        type=str,
        help="The tokenizer used for evaluation.",
    )

    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    files = [f for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f)) and f.endswith('.jsonl')]
    vanilla_files = [f for f in files if 'vanilla' in f]
    if len(vanilla_files) != 1:
        print(f"Error: Expected exactly one file containing 'vanilla' in the name, found {len(vanilla_files)}: {vanilla_files}")
        sys.exit(1)
    vanilla_path = os.path.join(args.folder, vanilla_files[0])

    # Subtasks to evaluate
    subtasks = ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag", "overall"]

    # Collect results
    results = {}
    mean_accepted_tokens = {}

    # Baseline: speedup is always 1.0, need to compute mean accepted tokens
    _, _, _, accept_lengths_list = speed(
        jsonl_file=vanilla_path,
        jsonl_file_base=vanilla_path,
        tokenizer=args.tokenizer,
        task="overall",
        report=False
    )
    results[vanilla_files[0]] = {subtask: 1.0 for subtask in subtasks}
    mean_accepted_tokens[vanilla_files[0]] = float(np.mean(accept_lengths_list))

    # Evaluate each system
    for f in files:
        if f == vanilla_files[0]:
            continue
        file_path = os.path.join(args.folder, f)
        results[f] = {}
        for subtask in subtasks:
            _, _, speedup_ratio, accept_lengths_list = speed(
                jsonl_file=file_path,
                jsonl_file_base=vanilla_path,
                tokenizer=args.tokenizer,
                task=subtask,
                report=False
            )
            results[f][subtask] = speedup_ratio
            if subtask == "overall":
                mean_accepted_tokens[f] = float(np.mean(accept_lengths_list))

    # Sort systems by descending overall speedup
    sorted_systems = sorted(results.items(), key=lambda x: x[1]["overall"], reverse=True)

    # Find common prefix for all .jsonl files
    def common_prefix(strings):
        if not strings:
            return ''
        s1 = min(strings)
        s2 = max(strings)
        for i, c in enumerate(s1):
            if i >= len(s2) or c != s2[i]:
                return s1[:i]
        return s1
    file_prefix = common_prefix([f for f in files])

    def clean_system_name(filename):
        # Remove prefix
        name = filename[len(file_prefix):] if filename.startswith(file_prefix) else filename
        # Remove everything from first hyphen
        hyphen_idx = name.find('-')
        if hyphen_idx != -1:
            name = name[:hyphen_idx]
        # Remove .jsonl extension if present
        if name.endswith('.jsonl'):
            name = name[:-6]
        return name

    # Print markdown table
    header = "| System | #Mean Accepted Tokens | " + " | ".join(subtasks) + " |"
    sep = "|---" * (len(subtasks) + 2) + "|"
    table_lines = []
    table_lines.append("\nMarkdown Table of Speedup Ratios:\n")
    table_lines.append(header)
    table_lines.append(sep)
    for system, subtask_dict in sorted_systems:
        system_name = clean_system_name(system)
        row = f"| {system_name} | {mean_accepted_tokens[system]:.2f} | " + " | ".join(f"{subtask_dict[subtask]:.3f}" for subtask in subtasks) + " |"
        table_lines.append(row)

    # Save to markdown file in the input folder
    output_path = os.path.join(args.folder, "speedup_table.md")
    with open(output_path, "w") as f:
        f.write("\n".join(table_lines) + "\n")
    print(f"\nMarkdown table saved to {output_path}\n")