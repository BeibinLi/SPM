from utils import (find_all_substr, slice_text)
from data_gen.paths import (prompt_output_path, raw_data_path,
                            chatlog_output_path, prompt_template_path)
from curious_agent import CuriousAgent
from gpt_api import get_llm

import os
import ast
from tqdm import tqdm
from collections import defaultdict
import tiktoken

from termcolor import colored

num_response = 2
num_interaction = 5
max_token_length = 10000
encoder = tiktoken.encoding_for_model("gpt-4")

tasks = {}
files = {}
total_data_count = defaultdict(int)
prompt_files = []


def save_prompt(prompt, task):
    if len(encoder.encode(prompt)) > max_token_length:
        return

    global total_data_count
    file_name = task + "_" + str(total_data_count[task]) + ".txt"
    path = prompt_output_path + file_name
    with open(path, 'w', encoding="utf-8") as handle:
        handle.write(prompt)
    total_data_count[task] += 1
    prompt_files.append(file_name)


def dfs(path):
    file_list = os.listdir(path)
    for file in file_list:
        if file.startswith("."):
            continue
        new_path = path + file
        if os.path.isdir(new_path):
            dfs(new_path + "/")
        else:
            with open(new_path, mode="r", encoding="utf-8") as handle:
                content = handle.read()
                if content.replace(" ", "").replace(
                        "\n", "") != "":    # remove empty files
                    files[new_path[len(raw_data_path):]] = content


def enumerate_file_tuples(file_names, content, pos, task):
    if pos == []:
        save_prompt(content, task)
        return

    for i in range(len(file_names)):
        pos_ed = pos[-1]
        while True:
            pos_ed += 1
            if content[pos_ed] == "}":
                break

        slices = slice_text(files[file_names[i]])

        for slice in slices:
            new_content = content[:pos[-1]] + slice + content[pos_ed + 1:]

            enumerate_file_tuples(file_names[i + 1:], new_content, pos[:-1],
                                  task)


def replace_content(file_names, task, prompt_template, identifier, suffix):
    # Replace identifiers in the template with files from file_names
    pos = find_all_substr(prompt_template, identifier)
    if pos == []:
        return

    filtered_fn = []
    for file_name in file_names:
        if file_name.endswith(suffix):
            filtered_fn.append(file_name)
    enumerate_file_tuples(filtered_fn, prompt_template, pos,
                          task[:-len(suffix)])


def extract_clip(code, clip_type):
    lines = code.split("\n")
    module = ast.parse(code)
    ret = []
    for node in module.body:
        if (clip_type == "class" and isinstance(node, ast.ClassDef)) \
            or (clip_type == "function" and isinstance(node, ast.FunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno
            ret.append("\n".join(lines[start:end]))
    return ret


def gen_code_prompts(file_names, task, prompt_template, clip_type):
    # TODO: implement multi {code_class} support
    pos = find_all_substr(prompt_template, "{code_" + clip_type + "}")
    if len(pos) != 1:
        return
    for file_name in file_names:
        if not file_name.endswith(".py"):
            continue
        code = files[file_name]
        class_clips = extract_clip(code, clip_type)
        for class_clip in class_clips:
            content = prompt_template
            pos_ed = pos[0]
            while True:
                pos_ed += 1
                if content[pos_ed] == "}":
                    break
            content = content[:pos[0]] + class_clip + content[pos_ed + 1:]

            save_prompt(content, task)


def gen_data(data_type):
    dfs(raw_data_path + data_type + "/")
    file_names = list(files.keys())

    for (task, prompt_template) in tasks.items():
        print(colored(f"Generating {task}...", "green"))
        # All {content} {content%x} replacement
        # TODO: {prev_generated_QA} in question.md (all hybrid prompt templates)
        replace_content(file_names, task, prompt_template, "{content", ".md")

        # All {code} replacement
        replace_content(file_names, task, prompt_template, "{code}", ".py")

        # {code_class} or {code_function}
        for clip_type in ["class", "function"]:
            gen_code_prompts(file_names, task, prompt_template, clip_type)


if __name__ == "__main__":
    os.makedirs(prompt_output_path, exist_ok=True)
    os.makedirs(chatlog_output_path, exist_ok=True)

    file_list = os.listdir(prompt_template_path)
    for file in file_list:
        if not file.endswith(".md"):
            continue
        with open(prompt_template_path + file, mode="r") as handle:
            tasks[file] = handle.read()

    #for data_type in ["IFS_code", "IFS_document"]:
    for data_type in ["./"]:
        files = {}
        gen_data(data_type)

    for file in tqdm(prompt_files):
        store_path = chatlog_output_path + file[:-len(".txt"
                                                     )] + "_chatlog.pickle"
        if not os.path.exists(store_path):
            with open(prompt_output_path + file, "r") as handle:
                prompts = handle.read()
            agent = CuriousAgent(api=get_llm(),
                                 system_msg=prompts,
                                 formatter=None,
                                 temperature=1,
                                 top_p=0.6,
                                 num_response=num_response,
                                 max_token_length=max_token_length)

            for i in range(num_interaction):
                agent.reply()

            agent.dump(store_path)
