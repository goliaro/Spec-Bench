import os

#!/usr/bin/env python3

# Directory containing the input files
INPUT_DIR = "./data/cortex/model_answer/partial"
# Directory to write the aggregated files
OUTPUT_DIR = "./data/cortex/model_answer"
# Allowed prefixes to remove; their order determines the ordering in aggregation
ALLOWED_PREFIXES = [
    "CATEGORIZATION",
    "FEATURE_EXTRACTION",
    "QUESTION_SUGGESTION",
    "SQL_FANOUT1",
    "SQL_FANOUT2",
    "SQL_FANOUT3",
    "SQL_COMBINE",
]

def aggregate_files():
    # Mapping of base file name to list of tuples (prefix, file path)
    groups = {}

    # List all files in the input directory
    for fname in os.listdir(INPUT_DIR):
        fpath = os.path.join(INPUT_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        # Check if the file name starts with any of the allowed prefixes followed by an underscore
        for prefix in ALLOWED_PREFIXES:
            token = f"{prefix}_"
            if fname.startswith(token):
                base_name = fname[len(token):]
                groups.setdefault(base_name, []).append((prefix, fpath))
                break  # Once matched, do not check for other prefixes

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # For each group, sort the list according to the order of prefixes in ALLOWED_PREFIXES
    for base_name, items in groups.items():
        items.sort(key=lambda item: (ALLOWED_PREFIXES.index(item[0]), os.path.basename(item[1])))
        out_path = os.path.join(OUTPUT_DIR, base_name)
        with open(out_path, 'w', encoding='utf-8') as outfile:
            for _, file_path in items:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    # outfile.write("\n")  # Separator between files
        print(f"Aggregated {len(items)} files into {out_path}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    aggregate_files()
