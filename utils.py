"""Utility functions"""
import json
import os
import random
import shlex
import string
from typing import List

import tiktoken
from termcolor import colored

# exit should always be the last
SUPPORTED_CMDS = [
    "cd", "ls", "cat", "head", "tail", "echo", "python", "pip", "id", "exit"
]
FULL_CMDS = SUPPORTED_CMDS + [
    "pwd",
    "mkdir",
    "rmdir",
    "touch",
    "rm",
    "cp",
    "mv",
    "less",
    "grep",
    "find",
    "who",
    "w",
    "ps",
    "top",
    "kill",
    "tar",
    "chmod",
    "chown",
    "df",
    "du",
    "ifconfig",
    "ping",
    "netstat",
    "ssh",
    "scp",
]


def list_all_actions(root: str,
                     curr_dir: str,
                     allowed_file_exts: List[str] = None) -> List[str]:
    """
    Generate a list of actions to navigate through directories and read files
    in a directory tree.

    Args:
    - `root` (str): the root directory (absolute path) of the whole project.
    - `curr_dir` (str): The current directory (absolute path) to start the
    action list.
    - `allowed_file_exts` (List[str]): List of allowed file extensions to
        include in the action list.
    - `shuffle` (bool): should we shuffle the actions. (Default to True)

    Returns:
    - `action_list` (List[str]): A list of strings, each representing a
        "cd" or "cat" command.
    """
    assert curr_dir.startswith(root), (
        f"Coding Error: Current directory `{curr_dir}` not under root `{root}`."
    )

    action_list = ["cd .."] if root != curr_dir else []
    action_list.append("exit")

    # List all files and directories in the current directory
    entries = os.listdir(curr_dir)

    for entry in entries:
        entry_path = os.path.join(curr_dir, entry)

        # If the entry is a directory, navigate into it and explore
        # its contents recursively
        if os.path.isdir(entry_path):
            action_list.append(f"cd {wrap_path(entry)}")
            # action_list.extend(
            # list_all_actions(entry_path, allowed_file_exts))
            # action_list.append("cd ..")

        # If the entry is a file, read it using the "cat" command or identify
        # it using the "id" command if its extension is allowed
        elif os.path.isfile(entry_path):
            _, ext = os.path.splitext(entry)
            if allowed_file_exts is None or ext in allowed_file_exts:
                action_list.append(f"cat {wrap_path(entry)}")
                action_list.append(f"id {wrap_path(entry)}")

    return action_list


def optimal_action(curr_dir: str, target_file: str, actions: List[str]) -> int:
    """
    Find the index of the optimal action that leads to reading a target file.

    Args:
        curr_dir (str): The root directory where the actions start.
        target_file (str): The full path of the target file to be read.
        actions (List[str]): List of actions ('cd ...', 'cat ...') returned from
            the previous function.

    Returns:
        int: The index of the action that corresponds to reading the target
            file, or -1 if not found.
    """
    # Normalize path
    target_file = os.path.abspath(target_file)
    curr_dir = os.path.abspath(curr_dir)
    target_file = target_file.replace("\\", "/")    # windows format
    curr_dir = curr_dir.replace("\\", "/")    # windows format

    if not target_file.startswith(curr_dir):
        optimal_action = "cd .."
    elif os.path.dirname(target_file) == curr_dir:
        optimal_action = f"cat {os.path.basename(target_file)}"
    else:
        next_lvl = os.path.dirname(target_file).replace(curr_dir, "")
        next_lvl = next_lvl.split("/")
        next_lvl = [x for x in next_lvl if x != ""][0]
        optimal_action = f"cd {next_lvl}"

    assert optimal_action in actions, (
        f"Coding Error: Optimal action `{optimal_action}` not in action list.")
    return actions.index(optimal_action)


# Common programming language suffixes
CODE_SUFFIXES = (".py", ".c", ".cpp", ".cxx", ".cc", ".h", ".hpp", ".hxx",
                 ".cs", ".java", ".go")

# Common data file suffixes
DATA_SUFFIXES = (".csv", ".tsv", ".json")

# Common text file suffixes
TEXT_SUFFIXES = (".txt", ".md")


def list_files(directory: str, ignore_hidden: bool = True) -> list:
    """
    List all files in a directory recursively.

    Args:
    - `directory` (str): The path to the directory to list files from.
    - `ignore_hidden` (bool, optional): Whether to ignore hidden files.
        Defaults to True.

    Returns:
    - list: A list of file paths relative to the input directory.
    """

    def generator():
        for root, dirs, files in os.walk(directory):
            if ignore_hidden:
                dirs[:] = [d for d in dirs if not d.startswith('.')]

            for file in files:
                if ignore_hidden and file.startswith("."):
                    continue
                yield os.path.relpath(os.path.join(root, file), directory)

    return list(generator())


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
    # This pattern assumes that paths start with / followed by
    # any non-space characters.
    text = text.replace(root, "")
    text = text.replace(root[:-1], ".")
    return text


def display_files_recursively(
    folder_path: str,
    indent: str = "",
    file_suffixes: list = [".py", ".cpp", ".cs", ".md", ".txt", ".csv"],
    depth: int = 1,
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

    if depth > 1:
        # Recurse into directories
        for dir_name in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, dir_name)
            if os.path.isdir(dir_path):
                # Recursively check if sub-directory contains valid files or
                # folders with valid files
                ret += display_files_recursively(dir_path, indent + "    ",
                                                 file_suffixes, depth - 1)

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
        return _trunc_text(content, 1000, 500)
    elif file.endswith(DATA_SUFFIXES):
        return _trunc_text(content, 5, 500)
    elif file.endswith(TEXT_SUFFIXES):
        return _trunc_text(content, 100, 500)
    else:
        return _trunc_text(content, 10, 500)


def get_file_names(command: list) -> list:
    """
    Extract file names from the command.
    The file names are relative to the current working directory, so may need
    handling outside this function.

    Args:
    - command (list): The command splitted into a list.

    Returns:
    - list: A list of file names.
    """
    # remove '|'
    for i in range(len(command)):
        if command[i] == "|":
            command = command[:i]
            break

    if command[0] == "ls":
        if len(command) > 1:
            if command[1].startswith("-"):
                return ["."]
            return [command[1]]
        else:
            return ["."]
    elif command[0] == "cat":
        ret = [command[1]]
        if ">" in command or ">>" in command:
            ret.append(command[-1])
        return ret
    elif command[0] in ["head", "tail"]:
        return [command[-1]]
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


def extract_command_blocks(response: str,
                           identifier: str = "```bash",
                           only_first: bool = False) -> (list, list):
    """
    Extracts command blocks encapsulated by markdown code blocks from a given
    response string.

    Args:
    - `response` (str): The input string containing the bash commands enclosed
    in markdown code blocks.
    - `identifier` (str, optional): The identifier used to recognize the start
        of the bash commands block. Defaults to "\`\`\`bash", which can also be
        "\`\`\`python"
    - `only_first` (bool): Whether to only extract the first command block.

    Returns:
    - tuple: A tuple of two lists.
        - list: A list of strings, containing the extracted commands.
        Each command is a separate string in the list.
        - list: A list of tuple of two integers, indicating the start and end
        positions of the command blocks (including the identifier).

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

    if only_first:
        positions = positions[:1]

    end_positions = []
    for pos in positions:
        st = pos + len(identifier)
        p = response[st:].find("```") + st
        commands.append(response[st:p].strip())
        end_positions.append(p + 3)
    return commands, list(zip(positions, end_positions))


def split_command(command_block: str) -> list:
    """
    Split a command block into a list of arguments.

    Args:
    - command_block (str): A command block.

    Returns:
    - list: A list of arguments, with `None` denoting a newline.

    Example:
    Given the command block:
        echo "Hello, World!"
        cat file.txt
    The function will return:
        ['echo', '"Hello, World!"', None, 'cat', 'file.txt']
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
    _split = shlex.split(command_block)
    split = []

    j = 0
    for i in range(len(_split)):
        newline = False
        while command_block[j] != _split[i][0]:
            if command_block[j] == "\n":
                newline = True
            j += 1

        assert command_block[j:j + len(_split[i])] == _split[i]

        if newline:
            split.append(None)
        split.append(_split[i])
        j += len(_split[i])

    # Restore the quoted texts
    for i in range(len(split)):
        if split[i] is None:
            continue

        for j in range(len(split[i]) - L, -1, -1):
            substr = split[i][j:j + L]
            if substr in replacement_dict.keys():
                split[i] = split[i][:j] + replacement_dict[substr] + split[i][
                    j + L:]

    return split


def extract_commands(response: str, only_first: bool = False) -> list:
    """
    Parse a LLM output to a list of commands, where
    each command is represented in a list of arguments (strs).

    Args:
    - `response` (str): LLM's response.
    - `only_first` (bool): Whether to only extract the first command.

    Returns:
    - list: a 2D list of commands.
    """
    command_blocks = extract_command_blocks(response, only_first=only_first)[0]

    parsed_commands = []

    last_keyw_pos = 0
    for command_block in command_blocks:
        split = split_command(command_block)
        newline = True
        for i in range(len(split)):
            if split[i] is None:
                newline = True
            if (split[i] in FULL_CMDS and newline
                    and (i == 0 or (i > 0 and split[i - 1] != "|"))):
                parsed_commands.append(split[last_keyw_pos:i])
                last_keyw_pos = i
                newline = False
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

    if only_first:
        ret = ret[:1]

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
    assert command[0] == "echo", ("The command is not an echo command.")

    if command[1] == '-e':
        command = command[:1] + command[2:]

    for i in range(len(command)):
        if command[i].strip().startswith(">"):
            assert i == len(command) - 2
            return [
                "echo", " ".join(command[1:i]), command[i].strip(), command[-1]
            ]
    return ["echo", " ".join(command[1:])]


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


def wrap_path(filename: str) -> str:
    """
    Add quotes around a path if it contains spaces.

    Args:
    - `filename` (str): The path to wrap.

    Returns:
    - str: The wrapped path.
    """
    if " " in filename:
        filename = f'"{filename}"'
    return filename


def unwrap_path(filename: str) -> str:
    """
    Remove the quotes around a path if possible.

    Args:
    - `filename` (str): The path to unwrap.

    Returns:
    - str: The unwrapped path.
    """
    if not filename:
        return filename
    if filename[0] in ["\'", "\""] and filename[0] == filename[-1]:
        return unwrap_path(filename[1:-1])
    return filename


def load_dataset(task_file: str) -> list:
    if task_file.endswith(".json"):
        task_files = [task_file]
    else:
        task_files = [
            os.path.join(task_file, f)
            for f in os.listdir(task_file)
            if f.endswith(".json")
        ]
    dataset = []
    for task_file in task_files:
        dataset += json.load(open(task_file, "r"))

    ### Temporarily filter hidden files
    _dataset = []
    for data in dataset:
        if not data["filename"].startswith("."):
            _dataset.append(data)
    dataset = _dataset

    return dataset


def build_curriculum(dataset: list) -> list:
    """
    Build a curriculum for the dataset.
    The curriculum increases in the depth of the target file.

    Args:
    - `dataset` (list): The dataset.

    Returns:
    - list: The curriculum, which is a list of datasets with increasing depth.
    The later datasets do not contain the files in the previous datasets.
    """
    dataset = sorted(dataset, key=lambda x: x["filename"].count("/"))
    depth = -1
    dataset_by_depth = []
    for i in range(len(dataset)):
        cur_depth = dataset[i]["filename"].count("/")
        if cur_depth > depth:
            depth = cur_depth
            dataset_by_depth.append([dataset[i]])
        else:
            dataset_by_depth[-1].append(dataset[i])

    return dataset_by_depth
