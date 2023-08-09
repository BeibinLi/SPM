"""Utility functions"""
import os
import json
import random
from termcolor import colored
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from data_gen.paths import (
    pretrain_data_path,
    finetune_data_path,
    self_instruct_data_path,
    pretrain_raw_data_path,
)

def display_files_recursively(
    folder_path: str,
    indent: str = "",
    file_suffixes: list = [".py", ".cpp", ".cs", ".md", ".txt"],
) -> str:
    """Recursively lists files with specific suffixes from a given directory
      and its subdirectories.

    This function searches through the directory structure starting from
    `folder_path` and returns a string representation of the directory
    hierarchy containing only the files that match any of the provided
     `file_suffixes`. Each level in the hierarchy increases
    the indentation in the returned string, enhancing readability.

    Args:
    - folder_path (str): Path to the starting folder from which the
        search begins.
    - indent (str, optional): The indentation string for the current level
        of the directory hierarchy. Defaults to an empty string.
        Typically used internally for recursive calls.
    - file_suffixes (list of str, optional): A list of file suffixes
        (extensions) to consider while listing files. Defaults to [".py",
        ".cpp", ".cs", ".md", ".txt"].

    Returns:
    - str: A string representation of the directory hierarchy with files
        matching the provided suffixes. Directories are only displayed if they
        contain at least one valid file or subdirectory with valid files.

    Note:
    - The function assumes that the provided `folder_path` exists and is
        accessible.
    """
    ret = ""

    valid_files = [
        file_name for file_name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file_name))
        and file_name.endswith(tuple(file_suffixes))
    ]

    # Display valid files in the current folder
    for file_name in valid_files:
        if ret == "":
            ret += "\n" + indent + os.path.basename(folder_path) + "/"
        ret += "\n" + indent + "    " + file_name

    # Recurse into directories
    for dir_name in os.listdir(folder_path):
        dir_path = os.path.join(folder_path, dir_name)
        if os.path.isdir(dir_path):
            # Recursively check if sub-directory contains valid files or folders
            # with valid files
            ret += display_files_recursively(dir_path, indent + "    ",
                                             file_suffixes)

    return ret


def find_all_substr(string, substr):
    """
    Finds all occurrences of a substring in a string and returns their starting
    indices.

    This function scans the input string for all instances of the specified
    substring and returns a list of indices where these instances start. If the
    substring is not found in the string, an empty list is returned.

    Args:
    - string (str): The input string in which to search for the substring.
    - substr (str): The substring to search for.

    Returns:
    - list of int: A list of starting indices where the substring is found. The
         list is empty if the substring is not found.
    """
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
        ret = "|   " * (indention_level - 1) + "|-- "
    else:
        ret = ""
    ret += os.path.basename(path)

    if os.path.isdir(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if (not item.startswith(".")) and os.path.isdir(item_path):
                ret += "\n" + get_directory_tree(item_path, indention_level + 1)

    return ret


def colored_string(msg: str) -> str:
    color_dict = {"system": "blue", "user": "green", "assistant": "cyan"}
    return colored(msg[1], color_dict[msg[0]])


def get_exp_id(result_path: str) -> str:
    """
    Retrieves the next experiment ID based on existing experiment directories.

    This function scans the directory specified by `ckpt_path` to identify
    existing experiment directories. The experiment ID is derived from the first
    three characters of directory names, which are expected to be integers.
    The function then increments the highest found experiment ID by one to
    generate the next ID. If no existing experiment directories are found,
    the function returns an ID of "000".

    Args:
    - ckpt_path (str): Path to the directory containing experiment directories.

    Returns:
    - str: The next experiment ID, zero-padded to ensure a length of three
        characters.

    Note:
    Any directories that do not conform to the expected naming convention (i.e.,
        directories without an integer as the first three characters) are
        ignored.
    """
    os.makedirs(result_path, exist_ok=True)
    exp_dirs = os.listdir(result_path)
    exp_num_list = []
    for x in exp_dirs:
        try:
            exp_num_list.append(int(x[:3]))
        except Exception:
            print("Unable to fetch the experiment number for directory: " + x)
    exp_id = max(exp_num_list) + 1 if exp_num_list != [] else 0
    return str(exp_id).zfill(3)


def get_spm_dataset(phase: str,
                    mode: str,
                    with_self_instruct: bool = False) -> Dataset:
    """Retrieves the specified dataset based on the provided phase and mode,
     with an optional inclusion of self-instructed data.

    This function loads datasets based on the given phase (e.g., 'baseline',
    'pretrain', 'finetune') and mode (e.g., 'train', 'test'). Depending on the
    provided phase, different data paths are chosen, and if
    `with_self_instruct` is set to True, the self-instructed data path is also
    considered. However, self-instructed data is not used in the 'finetune'
     phase.

    Args:
    - phase (str): The phase of the data. Expected values are 'baseline',
        'pretrain', or 'finetune'.
    - mode (str): The mode of the data. Expected values are 'train' or 'test'.
    - with_self_instruct (bool, optional): Whether to include self-instructed
        data. Defaults to False.

    Returns:
    - Dataset: The specified dataset loaded from the chosen data files, shuffled
         with a seed of 42.

    Raises:
    - ValueError: If an invalid mode or phase is provided.

    Note and TODO:
    The actual paths like 'pretrain_raw_data_path', 'self_instruct_data_path',
        etc., are assumed to be available in the current scope. Make it more
        flexible and formal.
    """

    if mode not in ["train", "test"]:
        raise ValueError("Invalid mode: " + mode +
                         ". Valid modes are: {train, test}")

    if phase == "baseline":
        data_files = [
            pretrain_raw_data_path + mode + ".jsonl",
        ]
        if with_self_instruct:
            data_files.append(self_instruct_data_path + mode + ".jsonl")
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
            print(
                colored(
                    "Warning: self-instructed data is not used in finetune phase",
                    "yellow",
                ))
    else:
        raise ValueError("Invalid phase: " + phase +
                         ". Valid phases are: {baseline, pretrain, finetune}")

    return load_dataset("json", data_files=data_files,
                        split="train").shuffle(seed=42)


def save_data(data: dict,
              train_path: str,
              test_path: str,
              train_percent: float = 0.7) -> None:
    """
    Splits the input data into training and testing datasets and saves them to
    the specified paths.

    The function first identifies the unique values from the input data. It then
    samples 70% of these unique values to form the training set. The keys
    corresponding to these sampled values form the training data. The rest form
    the testing data. Each dataset is saved in JSON format where each line
    corresponds to an entry with a "text" key.

    Args:
    - data (dict): The input data dictionary. Expected to have a structure where
        keys correspond to textual data and values are some identifiers.
    - train_path (str): The path to the file where the training data should be
        saved.
    - test_path (str): The path to the file where the testing data should be
        saved.
    - train_percent (float): the percentage of training data. Default to 0.7

    Returns:
        None
    """
    all_values = set(list(data.values()))

    random.seed(1)
    train_set = random.sample(list(all_values),
                              int(len(all_values) * train_percent))

    train_data = [k for k, v in data.items() if v in train_set]

    test_data = [k for k, v in data.items() if v not in train_set]

    def dump(data: list, filename: str):
        with open(filename, "w") as f:
            f.write("\n".join([json.dumps({"text": d}) for d in data]))

    dump(train_data, train_path)
    dump(test_data, test_path)
