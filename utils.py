"""Utility functions"""
import os
import json
import random
from termcolor import colored
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
import shlex
import string
import tiktoken
from data_gen.paths import (
    pretrain_data_path,
    finetune_data_path,
    self_instruct_data_path,
    pretrain_raw_data_path,
)

# exit should always be the last
SUPPORTED_CMDS = [
    "cd", "ls", "cat", "head", "tail", "echo", "python", "pip", "exit", "id"
]

# Common programming language suffixes
CODE_SUFFIXES = (".py", ".c", ".cpp", ".cxx", ".cc", ".h", ".hpp", ".hxx",
                 ".cs", ".java", ".go")

# Common data file suffixes
DATA_SUFFIXES = (".csv", ".tsv", ".json")

# Common text file suffixes
TEXT_SUFFIXES = (".txt", ".md")


def list_files(directory: str, ignore_hidden: bool = True) -> list:
    """
    List all files in a directory (recursively).

    Args:
    - directory (str): The path to the directory to list files from.
    - ignore_hidden (bool, optional): Whether to ignore hidden files.
        Defaults to True.

    Returns:
    - list of str: A list of file paths relative to the input directory.
    """
    for root, dirs, files in os.walk(directory):
        if ignore_hidden:
            dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            if ignore_hidden and file.startswith("."):
                continue
            yield os.path.relpath(os.path.join(root, file), directory)


def hide_root(text, root) -> str:
    """
    Hide all root paths in a text.

    Args:
    - text (str): The text to replace paths in.
    - root (str): The root path.

    Returns:
    - str: The text with all root paths hidden.
    """
    # Regular expression pattern to match absolute file paths.
    # This pattern assumes that paths start with / followed by any non-space characters.
    text = text.replace(root, "")
    text = text.replace(root[:-1], ".")
    return text


def display_files_recursively(
    folder_path: str,
    indent: str = "",
    file_suffixes: list = [".py", ".cpp", ".cs", ".md", ".txt", ".csv"],
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


def trunc_text(file: str, content: str) -> str:
    """
    Truncate the content of a file for `cat`, `head`, `tail` command if it is too long.
    It will truncate to a maximum line and a maximum token, depending on the file type.

    Args:
    - file_name (str): The name of the file.
    - content (str): The content of the file.

    Returns:
    - str: The truncated content.
    """

    # Define truncate function
    def _trunc_text(content: str, max_line: int, max_token: int) -> str:
        """
        Truncate the content of a file for `cat`, `head`, `tail` command
        if it is too long.
        Truncate to `max_line` lines or `max_token` tokens, whichever is smaller.

        Args:
        - file_name (str): The name of the file.
        - content (str): The content of the file.
        - max_line (int): The maximum number of lines to display.
        - max_token (int): The maximum number of tokens to display.

        Returns:
        - str: The truncated content.
        """
        truncated = False

        lines = content.split("\n")
        if len(lines) > max_line:
            content = "\n".join(lines[:max_line])
            truncated = True

        encoder = tiktoken.encoding_for_model("gpt-4")
        encoded = encoder.encode(content)
        if len(encoded) > max_token:
            content = encoder.decode(encoded[:max_token])
            truncated = True

        if truncated:
            content += ("\n...\nLarge file, only display first "
                        f"{max_line} lines and {max_token} tokens.\n")

        return content

    if file[-1] in ['"', "'"]:
        file = file[1:-1]

    # Truncate the content depending on file type
    if file.endswith(CODE_SUFFIXES):
        return _trunc_text(content, 1000, 1000)
    elif file.endswith(DATA_SUFFIXES):
        return _trunc_text(content, 5, 500)
    elif file.endswith(TEXT_SUFFIXES):
        return _trunc_text(content, 100, 1000)
    else:
        return _trunc_text(content, 10, 1000)


def get_file_names(command: list) -> list:
    """
    Extract file names from the command.

    Args:
    - command (list): The command splitted into a list.

    Returns:
    - list: A list of file names.
    """
    if command[0] == "ls":
        if len(command) > 1:
            return [command[1]]
        else:
            return ["."]
    elif command[0] == "cat":
        ret = [command[1]]
        if ">" in command or ">>" in command:
            ret.append(command[-1])
        return ret
    elif command[0] in ["head", "tail"]:
        return [command[3]]
    elif command[0] == "cd":
        return [command[1]]
    elif command[0] == "echo":
        return [command[-1]]
    elif command[0] == "python":
        if command[2] == "-c":
            return ["."]
        else:
            for x in command:
                if x.endswith(".py"):
                    return [x]
    elif command[0] == "pip":
        return ["."]
    elif command[0] == "id":
        return [command[1]]
    else:
        raise NotImplementedError(f"Does not support command: {command[0]}")


def extract_command_blocks(response, identifier="```bash") -> list:
    """
    Extracts command blocks encapsulated by markdown code blocks from a given
    response string.

    Parameters:
    - response (str): The input string containing the bash commands enclosed in
        markdown code blocks.
    - identifier (str, optional): The identifier used to recognize the start
        of the bash commands block. Defaults to "```bash", which can also be
        "```python"

    Returns:
    - list: A list of strings, containing the extracted commands.
         Each command is a separate string in the list.

    Example:
    Given the response string:
    '''
    Some text here.
    ```bash
    echo "Hello, World!"
    ls
    ```
    Another text.
    ```bash
    pwd
    ```
    '''
    The function will return:
    ['echo "Hello, World!"\nls', 'pwd']

    Note:
    The function assumes that the end of a bash commands block is marked
    by "```".
    """
    commands = []
    positions = find_all_substr(response, identifier)
    for pos in positions:
        st = pos + len(identifier)
        p = response[st:].find("```") + st
        commands.append(response[st:p].strip())
    return commands


def split_command(command_block: str) -> list:
    """
    Split a command block into a list of arguments.

    Args:
    - command_block (str): A command block.

    Returns:
    - list: A list of arguments.

    Example:
    Given the command block:
        echo "Hello, World!"
        cat file.txt
    The function will return:
        ['echo', '"Hello, World!"', 'cat', 'file.txt']
    """
    indices = []
    quote = None

    # Find all quoted texts
    for i in range(len(command_block)):
        if command_block[i] in ["'", '"']:
            if i > 0 and command_block[i - 1] == "\\":
                # \' = '(single character) if outside quote
                # \' = \' if inside quote
                # \" = "(single character) any time
                if (command_block[i] == '"'
                        or (command_block[i] == "'" and quote is None)):
                    #i += 1
                    continue
            if quote is None:
                quote = command_block[i]
                pos = i
            elif quote == command_block[i]:
                quote = None
                indices.append((pos, i))
            # elif command_block[i] == '"':
            #     command_block = command_block[:i] + '\\' + command_block[i:]
            #     i += 1
        #i += 1

    # Replace quoted texts with random strings
    L = 10
    replacement_dict = {}
    for index in reversed(indices):
        text = command_block[index[0]:index[1] + 1]
        while True:
            replacement = ''.join(
                random.choices(string.ascii_letters + string.digits, k=L))

            if replacement in replacement_dict.values():
                continue

            if replacement in (command_block[:index[0]] + "@" +
                               command_block[index[1] + 1:]):
                continue

            break
        replacement_dict[replacement] = text
        command_block = (command_block[:index[0]] + replacement +
                         command_block[index[1] + 1:])

    # Replace escaped spaces with a random string
    while True:
        replacement = ''.join(
            random.choices(string.ascii_letters + string.digits, k=L))
        if replacement not in command_block:
            num_escaped_space = command_block.count("\\ ")
            _command_block = command_block.replace("\\ ", replacement)
            # Check if replacement creates unwanted substrings
            if _command_block.count(replacement) == num_escaped_space:
                break
    command_block = command_block.replace("\\ ", replacement)
    replacement_dict[replacement] = "\\ "

    # Split the command
    split = shlex.split(command_block)

    # Restore the quoted texts
    for i in range(len(split)):
        for j in range(len(split[i]) - L, -1, -1):
            substr = split[i][j:j + L]
            if substr in replacement_dict.keys():
                split[i] = split[i][:j] + replacement_dict[substr] + split[i][
                    j + L:]

    return split


def extract_commands(response: str) -> list:
    """
    Parse a LLM output to a list of commands, where
    each command is represented in a list of arguments (strs).

    Args:
    - response (str): LLM's response.

    Returns:
    - list: a 2D list of commands.
    """
    command_blocks = extract_command_blocks(response)

    parsed_commands = []

    last_keyw_pos = 0
    for command_block in command_blocks:
        split = split_command(command_block)
        for i in range(len(split)):
            if (split[i] in SUPPORTED_CMDS
                    and (i == 0 or (i > 0 and split[i - 1] != "|"))):
                parsed_commands.append(split[last_keyw_pos:i])
                last_keyw_pos = i
        parsed_commands.append(split[last_keyw_pos:])

    ret = []

    for cmd in parsed_commands:
        if cmd == []:
            continue
        if cmd[0] == "echo":
            cmd = parse_echo(cmd)
        elif cmd[0] == "python":
            # Ignore warnings
            cmd.insert(1, "-W ignore")
        ret.append(cmd)

    return ret


def parse_echo(command: list) -> list:
    """
    Parses an `echo` command list into 5 parts.

    The function groups the echo command arguments into its main components,
    specifically handling redirection using the '>' symbol.

    Args:
    - command (list): A list of strings representing the `echo` command split
        by whitespace.

    Returns:
    - list: A list of parsed components.

    Example:
    Given the command list:
    ['echo', 'Hello', 'World', '>', 'output.txt']
    The function will return:
    ['echo', '-e', '"Hello World"', '>', 'output.txt']

    Note:
    The function assumes that the redirection symbol '>' is always followed by
        the filename
    and that the redirection symbol will only appear once at the end of the
        command.
    The `"` characters are added to the message to be echoed to ensure that
        the message is encapsulated. Only run in Linux.
    """
    assert command[0] == "echo", "The command is not an echo command."

    if command[1] == '-e':
        command = command[:1] + command[2:]

    for i in range(len(command)):
        if command[i].strip().startswith(">"):
            assert i == len(command) - 2
            return [
                "echo", " ".join(command[1:i]), command[i].strip(), command[-1]
            ]
    return ["echo", " ".join(command[1:])]


def get_target_dirs(cmd: list) -> list:
    """
    Get the directory of the target file/dir from a command.

    Args:
    - cmd (list): a single command splitted into a list of arguments.

    Returns:
    - list: A list of the directories of the target file/dirs.
            If error occurs, return a list of error messages.
    """
    # Get the files
    files = get_file_names(cmd)
    target_dirs = []

    for file in files:
        path = os.path.dirname(file) if "." in os.path.basename(file) else file
        if path == "":
            path = "."

        # Backup the cwd
        original_cwd = os.getcwd()

        try:
            os.chdir(path)
        except Exception as e:
            return ["Error: " + str(e)]

        target_dirs.append(os.getcwd().replace('\\', '/') + "/")

        # Restore the cwd
        os.chdir(original_cwd)

    return target_dirs


def slice_text(text: str,
               slicing_gap: int = 800,
               slicing_len: int = 1000) -> list:
    encoder = tiktoken.encoding_for_model("gpt-4")

    ret = []
    encoded = encoder.encode(text)
    i = 0
    while i < len(encoded):
        ret.append(encoder.decode(encoded[i:i + slicing_len]))
        i += slicing_gap

    return ret


def handle_ls(stdout: str) -> str:
    """
    Add quotes to the filenames after 'ls'.

    Args:
    - `stdout` (str): The standard output of 'ls'.

    Returns:
    - str: The formatted output with quotes.
    """

    files = stdout.split("\n")
    return '\n'.join([f"'{x}'" for x in files if x != ""])
