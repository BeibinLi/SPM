import os
import json

from transformers import HfArgumentParser

from experiment_args import ScriptArguments


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

files = [f for f in os.listdir(script_args.task_file) if f.endswith(".json")]
total_length = 0
file_lengths = {}

repo_data = {}

# Calculate total data length
for file in files:
    file_path = os.path.join(script_args.task_file, file)
    data = read_json_file(file_path)
    for d in data:
        repo = d['root']
        if repo not in repo_data:
            repo_data[repo] = []
        repo_data[repo].append(d)

for data in repo_data.values():
    length = len(data)    # Assuming data is a list
    total_length += length

# Calculate lengths for each split
lengths = {
    "train": total_length * 0.7,
    "validate": total_length * 0.2,
    "test": total_length, # all remaining data goes to test
}

# Distribute files
distributions = {'train': [], 'validate': [], 'test': []}
current_lengths = {'train': 0, 'validate': 0, 'test': 0}

for repo, data in repo_data.items():
    length = len(data)

    for split in ["train", "validate", "test"]:
        if current_lengths[split] + length <= lengths[split]:
            distributions[split].append(data)
            current_lengths[split] += length
            break

# Create folders and move files
for split, data in distributions.items():
    split_folder = os.path.join(script_args.task_file, split)
    os.makedirs(split_folder, exist_ok=True)
    for d in data:
        file_name = d[0]['root'] + '.json'
        file_path = os.path.join(split_folder, file_name)
        with open(file_path, 'w') as file:
            json.dump(d, file, indent=4)
