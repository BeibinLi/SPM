"""Generate exploration problem set (Chat history).
Find a file, write a question, and create a path"""
import glob
import json
import os
import pdb
import random
import re
import sys
from typing import List

from termcolor import colored

from gpt_utils import num_tokens, reply

sys.path.append(os.path.join(os.path.dirname(__file__), '../../SPM'))
from utils import list_files  # SPM's utils

MODEL_NAME = "gpt-4"
MAX_COMMAND_STEPS = 100


def _random_file(root: str, suffix: list, ignore_regex: list) -> str:
    # Initialize an empty list to store the filenames that
    # match the suffix criteria
    matching_files = []

    # Traverse through the directory tree starting from the root
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            # Check if the file has one of the defined suffixes
            if any(filename.endswith(s) for s in suffix):
                # The file extension matches one of the suffixes
                dirpath = dirpath.replace("\\", "/")
                folder_paths = dirpath.split("/")
                if any(f.startswith(".") for f in folder_paths):
                    # Skip: the file is in a hidden folder
                    continue

                fname = os.path.join(dirpath, filename)
                if any(re.search(regex, fname) for regex in ignore_regex):
                    # The file name matches one of the ignore regexes
                    continue
                # Append the full path of the file to the list
                matching_files.append(fname)

    # Randomly select a file from the list of matching files
    return random.choice(matching_files) if matching_files else None


def _random_block(filename: str, n_max_words: int) -> str:
    """
    Given a large file, return a random chunk of text (split by lines) 
    containing up to n_max_words words.
    
    Parameters:
        filename (str): The path to the file from which to read.
        n_max_words (int): The maximum number of words that the random block of 
            text should contain.
        
    Returns:
        str: A string containing the random block of text from the file.
    """
    # Initialize an empty list to store all lines from the file
    all_lines = []

    # Read all lines from the file and store them in the list
    with open(filename, 'r') as f:
        all_lines = f.readlines()

    # If the file is empty or contains no lines, return an empty string
    if not all_lines:
        return ""

    # Randomly choose a starting line index
    start_idx = random.randint(0, len(all_lines) - 1)

    # Initialize variables to keep track of the number of words and the
    # selected lines
    n_words = 0
    selected_lines = []

    # Loop to collect lines until n_max_words is reached or the end of
    # the file is reached
    for line in all_lines[start_idx:]:
        # line_words = line.split()
        n_words += num_tokens(line)

        if n_words > n_max_words:
            break

        selected_lines.append(line)

    return '\n'.join(selected_lines)


def norm_path(filename):
    # Check if the string contains any non-alphanumeric characters
    if any(not ch.isalnum() and ch not in [".", "_", "-", "/"]
           for ch in filename):
        return f'"{filename}"'
    return filename


def unnorm_path(filename):
    if filename[0] in ["\'", "\""] and filename[0] == filename[-1]:
        return unnorm_path(filename[1:-1])
    return filename


def commands_to_reach_destination(start: str,
                                  destination: str,
                                  folder_find_acc: float = 0.8) -> List[str]:
    """Use Linux's "ls", "cd", and "cat command to explore the path, 
    from `start` to the `destination`.

    Note that you are unfamiliar with the path, so you may need to "ls" to see
    the content inside a folder.

    It is guaranteed that the start and destination exist. 

    Args:
        start (str): a path
        destination (str): filename, a file which we want to find. 
        folder_find_acc (float, optional): the probability of finding a 
            correct folder. Defaults to 0.8.

    Returns:
        commands (list): a list of commands
    """
    assert os.path.isdir(start)
    assert os.path.isfile(destination)

    # Initialize an empty list to store the commands
    commands = []

    curr = start
    while True:
        commands.append("ls")

        # List the contents in the current directory
        contents = os.listdir(curr)

        # Separate the contents into files and folders
        files = [f for f in contents if os.path.isfile(os.path.join(curr, f))]
        folders = [f for f in contents if os.path.isdir(os.path.join(curr, f))]

        # Roll a dice to find the correct file
        if os.path.realpath(destination) in [
                os.path.realpath(os.path.join(curr, fname)) for fname in files
        ]:
            correct_command = f"cat {norm_path(os.path.basename(destination))}"
        else:
            # If we haven't found the file, we should try folders
            correct_path = os.path.relpath(destination, curr)
            cd_to = correct_path.split('/')[0]
            correct_command = f"cd {norm_path(cd_to)}"

        # List all possible actions
        all_possible_commands = [
            f"cat {norm_path(os.path.basename(fname))}" for fname in files
        ] + [f"cd {norm_path(os.path.basename(folder))}" for folder in folders]
        if curr != start:
            all_possible_commands += ["cd .."]

        if random.random() <= folder_find_acc:
            commands.append(correct_command)
        else:
            commands.append(random.choice(all_possible_commands))

        if commands[-1].startswith("cd "):
            dirname = unnorm_path(commands[-1][3:])
            curr = os.path.join(curr, dirname)
            curr = os.path.realpath(curr)
        else:
            cat_fname = unnorm_path(commands[-1].split()[-1])
            if os.path.realpath(os.path.join(
                    curr, cat_fname)) == os.path.realpath(destination):
                break

        if len(commands) > MAX_COMMAND_STEPS:
            raise RuntimeError("Too many commands")

    # Remove redundant "ls" command.
    # iterate through the commands, we don't need two adjacent "ls" commands,
    # even if there are some "cat" inside.
    rst = []
    need_ls = True
    for cmd in commands:
        if cmd.startswith("ls"):
            if need_ls:
                rst.append(cmd)
                need_ls = False
        else:
            rst.append(cmd)
            if cmd.startswith("cd"):
                need_ls = True

    print(colored(destination, "green"))
    print(commands)

    return commands


def optimal_path(start: str, destination: str) -> List[str]:
    """Use Linux's "ls", "cd", and "cat command to explore the path, 
    from `start` to the `destination`.

    Note that you are unfamiliar with the path, so you may need to "ls" to see
    the content inside a folder.

    It is guaranteed that the start and destination exist. 

    Args:
        start (str): a path
        destination (str): filename, a file which we want to find. 
        folder_find_acc (float, optional): the probability of finding a 
            correct folder. Defaults to 0.8.

    Returns:
        commands (list): a list of commands
    """
    assert os.path.isdir(start)
    assert os.path.isfile(destination)

    # folders = os.path.relpath(destination, start).split("/")
    # commands = []
    # for dirname in folders[:-1]:
    #     commands += ["ls", f"cd {dirname}"]
    # commands += ["ls", f"cat {folders[-1]}"]

    folders = os.path.relpath(destination, start).split("/")
    commands = [
        cmd for dirname in folders[:-1]
        for cmd in ["ls", f"cd {norm_path(dirname)}"]
    ] + ["ls", f"cat {norm_path(folders[-1])}"]

    return commands


def _random_question(root: str, filename: str, block: str) -> str:

    prompt = f"""
I found the following content in the file {filename}.

--- Repository Architecture ---
{list_files(root)}
------

Now, generate reading comprehension questions and answers based on the content.
You question should be very specific to the given file and the given content.
If the question is too general, don't even bother generating it.
The question should NOT be relevant to other files within the repository.

Use the format:

QUESTION: a question goes here
ANSWER: the answer to the question goes here

--- Good question example ---
How to generate an agent in the package?
Write code for "optimizer".
Show me suffix that are ignored from the pre-processing step.


--- Bad question example ---
What is the purpose of the code?
How to use the help function?
What is the purpose of @staticmethod?


--- Here are the content ---
{block}
"""
    open("prompt.tmp", "w").write(prompt)
    # pdb.set_trace()

    ans = reply(prompt, model_name=MODEL_NAME)

    pairs = re.findall("QUESTION: (.*?)ANSWER: (.+)", ans, re.DOTALL)
   
    if len(pairs) == 0:
        print(colored("NO questions generated by GPT!", "red"), filename)
        # pdb.set_trace()

    verified_pairs = []
    for question, answer in pairs:
        verify_prompt = f"""
Is the following question and answer specific to the given content? Or, is it a general question?

--- Here are the content ---
{block}

Question: {question}
Answer: {answer}

--- Reply with one word: SPECIFIC or GENERAL ---
"""
        ans_v = reply(verify_prompt, model_name=MODEL_NAME)

        if ans_v.upper().find("SPECIFIC") >= 0:
            verified_pairs.append((question, answer))
        else:
            print(colored("Skip the question:", "red"), question, "\t", answer)

    pdb.set_trace()
    return verified_pairs


def gen(root: str, n_files: int = 10, outname: str = "out.json"):
    if os.path.exists(outname):
        print(f"Skip {outname}, because it already exists~")
        return

    rst = []
    num_success_attempts = 0

    while num_success_attempts < n_files:
        filename = _random_file(
            root=root,
            suffix=FILES_SUFFIX,
            ignore_regex=[".*out.*", ".*\.git.*", ".*test.*"])

        try:
            block = _random_block(filename, 3000)
            if num_tokens(block) < 100:
                print("A small block ignored in:", filename)
                continue
            pairs = _random_question(root, filename, block)
        except:
            print(colored("Fail to generate questions for file: ", "red"),
                  filename)
            continue
        # assert len(pairs), "NO questions generated by GPT!"
        if len(pairs) == 0:
            # use empty answer
            # pairs = [("", "")]
            print(colored("NO questions generated by GPT!", "red"), filename)
        else:
            num_success_attempts += 1

        optimal = optimal_path(root, filename)  # the optimal path
        n_level = len([cmd for cmd in optimal if cmd.startswith("cd ")])

        for question, answer in pairs:
            try:
                commands = commands_to_reach_destination(root,
                                                         filename,
                                                         folder_find_acc=0.8)
            except Exception as e:
                print(e)
                continue

            print(colored(optimal, "red"))

            rst.append({
                "question": question,
                "answer": answer,
                "commands": commands,
                "optimal_path": optimal,
                "filename": os.path.relpath(filename, root),
                "root": os.path.relpath(root, REPOS_ROOT),
                "n_level": n_level
            })


    if len(rst) == 0:
        pdb.set_trace()


    # dump `rst` to json
    with open(outname, "w") as f:
        json.dump(rst, f, indent=2)


if __name__ == "__main__":
    FILES_SUFFIX = [
        '.txt', '.md', '.py', '.ipynb', '.swift', '.js', '.css', '.html',
        '.java', '.c', '.cpp', '.h', '.hpp', '.sh', '.bash', '.yml', '.json',
        '.xml', '.rb', '.go', '.php', '.rs', '.ts', '.vue', '.scss', '.sass',
        '.m', '.mm', '.pl', '.perl', '.lua', '.kt', '.kts', '.gradle', '.cs',
        '.fs', '.fsharp', '.vb', '.vbs', '.vba', '.r', '.R', '.dart', '.pas',
        '.p', '.pp', '.asm', '.s', '.lisp', '.clj', '.cljs', '.scala', '.hs',
        '.lhs', '.erl', '.hrl', '.ex', '.exs', '.elm', '.groovy', '.gvy', '.gy',
        '.gsh', '.mjs', '.cjs', '.ejs', '.coffee', '.litcoffee', '.iced',
        '.ini', '.toml', '.cfg', '.conf', '.properties', '.prop', '.sql',
        '.ps1', '.bat', '.cmd', '.vbscript', '.vbs', '.awk', '.make', '.mk',
        '.cmake', '.d', '.dockerfile', '.proto', '.twig', '.jl', '.mat'
    ]

    random.seed(1)

    # print(cmds)
    REPOS_ROOT = os.path.expanduser("~/data/repos")
    OUT_DIR = "rst_search"
    os.makedirs(OUT_DIR, exist_ok=True)
    for dirname in os.listdir(REPOS_ROOT):
        print(f"Processing {dirname}...")
        gen(os.path.join(REPOS_ROOT, dirname),
            n_files=10,
            outname=os.path.join(OUT_DIR, f"{dirname}.json"))

        # pdb.set_trace()

    # Combine all /*.json into one json file
    all_data = []
    for filename in glob.glob(os.path.join(OUT_DIR, "*.json")):
        all_data += json.load(open(filename, "r"))

    json.dump(all_data, open("file_search.json", "w"), indent=2)
