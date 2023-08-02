import os
from termcolor import colored
from datasets import load_dataset
from data_gen.paths import pretrain_data_path, finetune_data_path, self_instruct_data_path, pretrain_raw_data_path

def display_files_recursively(folder_path, indent='', 
                              file_suffixes=[".py", ".cpp", ".cs", ".md", ".txt"]):
    ret = ""
    
    valid_files = [file_name for file_name in os.listdir(folder_path) 
                   if os.path.isfile(os.path.join(folder_path, file_name))
                   and file_name.endswith(tuple(file_suffixes))]
    
    # Display valid files in the current folder
    for file_name in valid_files:
        if ret == "":
            ret += "\n" + indent + os.path.basename(folder_path) + '/'
        ret += "\n" + indent + '    ' + file_name

    # Recurse into directories
    for dir_name in os.listdir(folder_path):
        dir_path = os.path.join(folder_path, dir_name)
        if os.path.isdir(dir_path):
            # Recursively check if sub-directory contains valid files or folders with valid files
            ret += display_files_recursively(dir_path, indent + '    ', file_suffixes)

    return ret

def find_all_substr(string, substr):
    start_index = 0
    positions = []

    while True:
        index = string.find(substr, start_index)
        if index == -1:
            break
        positions.append(index)
        start_index = index + 1
    
    return positions

def get_directory_tree(path, indention_level=0):
    # add the '|' symbol before the folder name to represent levels
    if indention_level:
        ret = '|   ' * (indention_level - 1) + '|-- '
    else:
        ret = ""
    ret += os.path.basename(path)

    if os.path.isdir(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if (not item.startswith(".")) and os.path.isdir(item_path):
                ret += "\n" + get_directory_tree(item_path, indention_level + 1)
    
    return ret

def colored_string(msg):
    color_dict = {"system": "blue", "user": "green", "assistant": "cyan"}
    return colored(msg[1], color_dict[msg[0]])


def get_exp_id(ckpt_path):
    os.makedirs(ckpt_path, exist_ok=True)
    exp_dirs = os.listdir(ckpt_path)
    exp_num_list = [int(x) for x in exp_dirs if x.isdigit()]
    exp_id = max(exp_num_list) + 1 if exp_num_list != [] else 0
    return str(exp_id).zfill(3)

def get_spm_dataset(phase: str, mode: str, with_self_instruct: bool = False):
    '''
    phase: pretrain, finetune
    mode: train, test
    with_self_instruct: True, False. Whether to include self-instructed data
    '''
    if mode not in ["train", "test"]:
        raise ValueError("Invalid mode: " + mode + ". Valid modes are: {train, test}")
    
    if phase == "baseline":
        data_files = [
            pretrain_raw_data_path + mode + ".jsonl",
        ]
        if with_self_instruct:
            print(colored("Warning: self-instructed data is not used in baseline", "yellow"))
    elif phase == "pretrain":
        data_files = [
            pretrain_data_path + mode + ".jsonl",
        ]
        if with_self_instruct:
            data_files.append(self_instruct_data_path + mode + ".jsonl")
    elif phase == "finetune":
        data_files = [
            finetune_data_path + mode + ".jsonl",
        ]
        if with_self_instruct:
            print(colored("Warning: self-instructed data is not used in finetune phase", "yellow"))
    else:
        raise ValueError("Invalid phase: " + phase + ". Valid phases are: {baseline, pretrain, finetune}")
    
    return load_dataset("json", data_files=data_files, split="train").shuffle(seed=42)