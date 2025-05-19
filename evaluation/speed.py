import json
import argparse
import numpy as np
import os
import sys

def process_file(file_path):
    # Group stats by (question_id, category)
    results={}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            category = record["category"]
            question_id = record["question_id"]
            assert len(record["choices"]) == 1, f"Expected exactly one choice, got {len(record['choices'])}"
            
            latency = np.sum(record["choices"][0]["wall_time"])
            num_generated_tokens = np.sum(record["choices"][0]["new_tokens"])
            mat = np.mean(record["choices"][0]["accept_lengths"])
            assert num_generated_tokens > 0, f"num_generated_tokens should be greater than 0, got {num_generated_tokens}"
            tpot=latency/num_generated_tokens
            tpot_ms=tpot*1000
            key = (category, question_id)
            assert key not in results, f"Duplicate key {key} found in file {file_path}"
            results[key] = {
                "latency": latency,
                "mat": mat,
                "tpot": tpot_ms,
                "num_generated_tokens": num_generated_tokens,
            }
    return results

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


def clean_system_name(filename, file_prefix):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bench-name",
        type=str,
        required=True,
        choices=["spec_bench", "cortex", "swebench"],
        help="The folder containing evaluated Speculative Decoding methods and baseline.",
    )

    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    results_folder = os.path.join("./data", args.bench_name, "model_answer")

    files = [f for f in os.listdir(results_folder) if os.path.isfile(os.path.join(results_folder, f)) and f.endswith('.jsonl')]
    vanilla_files = [f for f in files if 'vanilla' in f]
    if len(vanilla_files) != 1:
        print(f"Error: Expected exactly one file containing 'vanilla' in the name, found {len(vanilla_files)}: {vanilla_files}")
        sys.exit(1)
    vanilla_path = os.path.join(results_folder, vanilla_files[0])
    file_prefix = common_prefix([f for f in files])
    
    # Process vanilla baseline
    vanilla_stats = process_file(vanilla_path)

    # Process all files and compute average speedup per file
    file_speedups = {}
    for filename in files:
        path = os.path.join(results_folder, filename)
        stats = process_file(path)
        speedups = []
        for key, values in stats.items():
            # Only compute speedup if the key is present in the vanilla baseline and baseline tpot is not zero
            if key in vanilla_stats and vanilla_stats[key]["tpot"]:
                speedups.append(vanilla_stats[key]["tpot"] / values["tpot"])
            else:
                print(f"Warning: Key {key} not found in vanilla stats or baseline tpot is zero.")
        # For vanilla baseline, define speedup as 1.0
        avg_speedup = 1.0 if filename in vanilla_files else (np.mean(speedups) if speedups else float("nan"))
        # print(f"filename {filename} speedups:", speedups)
        file_speedups[filename] = avg_speedup

    
    # Compute per file category aggregates for mat, tpot, and speedup
    all_categories = set()
    file_stats = {}  # Stores per-file aggregated metrics
    for filename in files:
        path = os.path.join(results_folder, filename)
        stats = process_file(path)
        per_cat = {}  # {category: {"mat": [...], "tpot": [...], "speedup": [...]}}
        overall = {"mat": [], "tpot": [], "speedup": []}
        for (category, question_id), values in stats.items():
            all_categories.add(category)
            if category not in per_cat:
                per_cat[category] = {"mat": [], "tpot": [], "speedup": []}
            per_cat[category]["mat"].append(values["mat"])
            per_cat[category]["tpot"].append(values["tpot"])
            if filename in vanilla_files:
                s_val = 1.0
            else:
                if (category, question_id) in vanilla_stats and vanilla_stats[(category, question_id)]["tpot"]:
                    s_val = vanilla_stats[(category, question_id)]["tpot"] / values["tpot"]
                else:
                    # Skip if vanilla data unavailable or baseline tpot is zero
                    continue
            per_cat[category]["speedup"].append(s_val)
            overall["mat"].append(values["mat"])
            overall["tpot"].append(values["tpot"])
            overall["speedup"].append(s_val)
        file_stats[filename] = {
            "per_cat": per_cat,
            "overall": {
                "mat": np.mean(overall["mat"]) if overall["mat"] else float("nan"),
                "tpot": np.mean(overall["tpot"]) if overall["tpot"] else float("nan"),
                "speedup": np.mean(overall["speedup"]) if overall["speedup"] else float("nan")
            }
        }

    categories_list = sorted(all_categories)

    # Helper to format a float value
    def fmt(val):
        return f"{val:.3f}" if not np.isnan(val) else "nan"

    # Build markdown tables for each metric
    def build_table(metric_name):
        header = "| System | " + " | ".join(categories_list) + " | Overall |"
        sep = "|" + "---|" * (len(categories_list) + 2)
        lines = [header, sep]

        # Determine sort order based on metric_name
        if metric_name in ["mat", "speedup"]:
            reverse_sort = True
        elif metric_name == "tpot":
            reverse_sort = False
        else:
            reverse_sort = False

        sorted_files = sorted(
            file_stats.keys(),
            key=lambda filename: file_stats[filename]["overall"][metric_name],
            reverse=reverse_sort,
        )

        for filename in sorted_files:
            system = clean_system_name(filename, file_prefix)
            row = [system]
            per_cat = file_stats[filename]["per_cat"]
            for category in categories_list:
                if category in per_cat and per_cat[category][metric_name]:
                    avg_val = np.mean(per_cat[category][metric_name])
                else:
                    avg_val = float("nan")
                row.append(fmt(avg_val))
            overall_val = file_stats[filename]["overall"][metric_name]
            row.append(fmt(overall_val))
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    table_mat = build_table("mat")
    table_tpot = build_table("tpot")
    table_speedup = build_table("speedup")

    md_content = "# Evaluation Metrics\n\n"
    md_content += "## Mean Accepted Tokens (tok/step):\n\n" + table_mat + "\n\n"
    md_content += "## TPOT (ms):\n\n" + table_tpot + "\n\n"
    md_content += "## Speedup (x):\n\n" + table_speedup + "\n\n"

    # Print markdown tables to console
    print(md_content)

    # Save the markdown file in the same folder where results are found
    output_md_file = os.path.join("./data", f"results_{args.bench_name}.md")
    with open(output_md_file, "w") as outfile:
        outfile.write(md_content)
    