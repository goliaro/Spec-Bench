from dataclasses import asdict, dataclass, field
from typing import List, Dict, Set
import json
import random
import os
from collections import defaultdict
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm

@dataclass
class TraceEntry:
    instance_id: str
    prompt: str
    response: str
    num_turns: int
    conversation_idx: int
    # prompt_length: int
    # response_length: int

@dataclass
class TracePartition:
    partition_name: str
    model_name: str
    training_entries: List[TraceEntry]
    eval_entries: List[TraceEntry]

# @dataclass
# class TraceMetadata:
#     avg_entries_per_partition: float
#     max_prompt_length: int
#     min_prompt_length: int
#     avg_prompt_length: float
#     max_response_length: int
#     min_response_length: int
#     avg_response_length: float

@dataclass
class Trace:
    partitions: List[TracePartition]
    # metadata: TraceMetadata = field(default_factory=lambda: TraceMetadata(0, 0, 0, 0, 0, 0, 0))

def get_trace(trace_file: str, model_name: str):
    # Load the tokenizer for Llama 3
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Path to the trace file
    trace_file = "./data/swebench_trace_incremental_decoding.jsonl"
    
    # Dictionary to store entries by repo
    entries_by_repo = defaultdict(list)
    instance_ids_by_repo = defaultdict(set)
    
    # Read the trace file
    with open(trace_file, 'r') as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, desc="Processing trace entries", total=total_lines):
            entry = json.loads(line)
            repo = entry["repo"]
            instance_id = entry["instance_id"]
            
            # Count user turns
            num_turns = sum(1 for msg in entry["messages"] if msg["role"] == "user")
            
            # Apply chat template for prompt using AutoTokenizer
            prompt = tokenizer.apply_chat_template(entry["messages"], add_generation_prompt=True, tokenize=False)
            
            # Get response content and create assistant message
            response_content = entry["response"]["content"]
            assert entry["response"]["role"] == "assistant"
            # response_message = [{"role": "assistant", "content": response_content}]
            
            # Apply chat template for response
            # response_with_template = tokenizer.apply_chat_template(response_message, add_generation_prompt=False, tokenize=False)
            
            # Count tokens
            # prompt_tokens = len(tokenizer.encode(prompt))
            # response_tokens = len(tokenizer.encode(response_with_template))
            
            # Create trace entry
            trace_entry = TraceEntry(
                instance_id=instance_id,
                prompt=prompt,
                response=response_content,
                num_turns=num_turns,
                conversation_idx=entry["idx"],
                # prompt_length=prompt_tokens,
                # response_length=response_tokens
            )
            
            entries_by_repo[repo].append(trace_entry)
            instance_ids_by_repo[repo].add(instance_id)
    
    # Create partitions
    partitions = []
    
    total_entries = 0
    all_prompt_lengths = []
    all_response_lengths = []
    
    for repo, entries in entries_by_repo.items():
        # Get unique instance_ids for this repo
        unique_instance_ids = list(instance_ids_by_repo[repo])
        print(f"Repo {repo} has {len(unique_instance_ids)} unique instance ids")
        
        # Determine how many instance_ids to use for evaluation
        num_eval_instances = min(2, len(unique_instance_ids))
        assert num_eval_instances > 0
        eval_instance_ids = set(random.sample(unique_instance_ids, num_eval_instances))
        
        # Split entries into training and evaluation
        training_entries = [e for e in entries if e.instance_id not in eval_instance_ids]
        eval_entries = sorted([e for e in entries if e.instance_id in eval_instance_ids], key=lambda x: x.num_turns)
        
        # Create partition
        partition = TracePartition(
            partition_name=repo,
            model_name=model_name,
            training_entries=training_entries,
            eval_entries=eval_entries
        )
        
        partitions.append(partition)
        
        # Gather statistics for metadata
        total_entries += len(entries)
        # all_prompt_lengths.extend([e.prompt_length for e in entries])
        # all_response_lengths.extend([e.response_length for e in entries])
    
    # Calculate metadata
    # avg_entries_per_partition = total_entries / len(partitions) if partitions else 0
    
    # metadata = TraceMetadata(
    #     avg_entries_per_partition=avg_entries_per_partition,
    #     max_prompt_length=max(all_prompt_lengths) if all_prompt_lengths else 0,
    #     min_prompt_length=min(all_prompt_lengths) if all_prompt_lengths else 0,
    #     avg_prompt_length=sum(all_prompt_lengths) / len(all_prompt_lengths) if all_prompt_lengths else 0,
    #     max_response_length=max(all_response_lengths) if all_response_lengths else 0,
    #     min_response_length=min(all_response_lengths) if all_response_lengths else 0,
    #     avg_response_length=sum(all_response_lengths) / len(all_response_lengths) if all_response_lengths else 0
    # )
    
    # Create the final trace
    trace = Trace(partitions=partitions)
    
    return trace

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./data/swebench_trace_incremental_decoding.jsonl", help="Input file path")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--output", type=str, default="./data/swebench/swebench_trace.json", help="Output file path")
    args = parser.parse_args()
    
    # Change to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    trace = get_trace(args.input, args.model)
    print(f"Created trace with {len(trace.partitions)} partitions")
    # print(f"Average entries per partition: {trace.metadata.avg_entries_per_partition:.2f}")
    # print(f"Prompt length: min={trace.metadata.min_prompt_length}, max={trace.metadata.max_prompt_length}, avg={trace.metadata.avg_prompt_length:.2f}")
    # print(f"Response length: min={trace.metadata.min_response_length}, max={trace.metadata.max_response_length}, avg={trace.metadata.avg_response_length:.2f}")

    # Convert the trace to a dictionary
    trace_dict = {
        "partitions": [asdict(partition) for partition in trace.partitions],
        # "metadata": asdict(trace.metadata)
    }
    
    # Write the trace to a JSON file
    with open(args.output, "w") as f:
        json.dump(trace_dict, f, indent=2)
    
    print(f"Trace saved to {args.output}")
    