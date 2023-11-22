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

# Calculate total data length
for file in files:
    file_path = os.path.join(script_args.task_file, file)
    data = read_json_file(file_path)
    length = len(data)    # Assuming data is a list
    file_lengths[file] = length
    total_length += length

# Calculate lengths for each split
train_length = total_length * 0.7
validate_length = total_length * 0.2
# Test length is the remaining data

# Sort files in some order if needed
# files.sort()

# Distribute files
distributions = {'train': [], 'validate': [], 'test': []}
current_lengths = {'train': 0, 'validate': 0, 'test': 0}

for file in files:
    length = file_lengths[file]
    if current_lengths['train'] + length <= train_length:
        distributions['train'].append(file)
        current_lengths['train'] += length
    elif current_lengths['validate'] + length <= validate_length:
        distributions['validate'].append(file)
        current_lengths['validate'] += length
    else:
        distributions['test'].append(file)
        current_lengths['test'] += length

# Create folders and move files
for split in distributions:
    os.makedirs(os.path.join(script_args.task_file, split), exist_ok=True)
    for file in distributions[split]:
        os.rename(os.path.join(script_args.task_file, file),
                  os.path.join(script_args.task_file, split, file))

# Your files are now distributed into 'train', 'validate', 'test' folders
