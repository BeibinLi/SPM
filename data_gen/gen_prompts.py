import os
import ast
from collections import defaultdict
import tiktoken

max_token_length = 3000
encoder = tiktoken.encoding_for_model("gpt-4")

prompt_path = "data_gen/prompt_templates/"
raw_data_path = "../raw_data/"
output_path = "data/"

prompts = {}
files = {}
total_data_count = defaultdict(int)

def save_content(path, content, prompt):
    if len(encoder.encode(content)) > max_token_length:
        return

    global total_data_count
    with open(path, 'w') as handle:
        handle.write(content)
    total_data_count[prompt] += 1

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

def dfs(path):
    file_list = os.listdir(path)
    for file in file_list:
        new_path = path + file
        if os.path.isdir(new_path):
            dfs(new_path + "/")
        else:
            with open(new_path, mode = "r") as handle:
                content = handle.read()
                if content.replace(" ", "").replace("\n", "") != "": # remove empty files
                    files[new_path[len(raw_data_path):]] = content

def enumerate_file_tuples(file_names, content, pos, file_path, prompt):        
    if pos == []:
        save_content(file_path + str(total_data_count[prompt]) + ".txt", content, prompt)
        return

    for i in range(len(file_names)):
        pos_ed = pos[-1]
        while True:
            pos_ed += 1
            if content[pos_ed] == "}":
                break
        
        new_content = content[:pos[-1]] + files[file_names[i]] + content[pos_ed+1:]

        enumerate_file_tuples(file_names[i+1:], new_content, pos[:-1], file_path, prompt)

def replace_content(file_names, prompt, template, identifier, suffix):
    # Replace identifiers in the template with files from file_names
    pos = find_all_substr(template, identifier)
    if pos == []:
        return
    
    filtered_fn = []
    for file_name in file_names:
        if file_name.endswith(suffix):
            filtered_fn.append(file_name)
    enumerate_file_tuples(filtered_fn, template, pos, output_path + prompt[:-3] + "_", prompt[:-3])

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

def gen_code_prompts(file_names, prompt, template, clip_type):
    # TODO: implement multi {code_class} support
    pos = find_all_substr(template, "{code_" + clip_type + "}")
    if len(pos) != 1:
        return
    for file_name in file_names:
        if not file_name.endswith(".py"):
            continue
        code = files[file_name]
        class_clips = extract_clip(code, clip_type)
        for class_clip in class_clips:
            content = template
            pos_ed = pos[0]
            while True:
                pos_ed += 1
                if content[pos_ed] == "}":
                    break
            content = content[:pos[0]] + class_clip + content[pos_ed+1:]

            save_content(output_path + prompt[:-3] + "_" + str(total_data_count[prompt]) + ".txt", content, prompt)

def gen_data(data_type):
    dfs(raw_data_path + data_type + "/")
    file_names = list(files.keys())

    for (prompt, template) in prompts.items():
        # All {content} {content%x} replacement
        # TODO: {prev_generated_QA} in question.md (all hybrid prompt templates)
        replace_content(file_names, prompt, template, "{content", ".md")

        # All {code} replacement
        replace_content(file_names, prompt, template, "{code}", ".py")

        # {code_class} or {code_function}
        for clip_type in ["class", "function"]:
            gen_code_prompts(file_names, prompt, template, clip_type)


if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)

    file_list = os.listdir(prompt_path)
    for file in file_list:
        if not file.endswith(".md"):
            continue
        with open(prompt_path + file, mode = "r") as handle:
            prompts[file] = handle.read()

    for data_type in ["IFS_code", "IFS_document"]:
        files = {}
        gen_data(data_type)
