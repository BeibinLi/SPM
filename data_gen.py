import os
import ast
from collections import defaultdict

prompt_path = "data_gen/prompts/"
raw_data_path = "data_gen/raw_data/"

prompts = {}
files = {}
total_data_count = defaultdict(int)

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
                files[new_path[len(raw_data_path):]] = handle.read()

def enumerate_file_tuples(file_names, content, pos, file_path, prompt):
    if pos == []:
        global total_data_count
        with open(file_path + str(total_data_count[prompt]) + ".txt", 'w') as handle:
            handle.write(content)
        total_data_count[prompt] += 1
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
    enumerate_file_tuples(filtered_fn, template, pos, "data/" + prompt[:-3] + "_", prompt[:-3])

def extract_class_info(code):
    lines = code.split("\n")
    module = ast.parse(code)
    ret = []
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            start = node.lineno - 1
            end = node.end_lineno
            ret.append("\n".join(lines[start:end]))
    return ret

def extract_code_class(file_names, prompt, template):
    # TODO: implement multi {code_class} support
    pos = find_all_substr(template, "{code_class}")
    if len(pos) != 1:
        return
    for file_name in file_names:
        if not file_name.endswith(".py"):
            continue
        code = files[file_name]
        class_clips = extract_class_info(code)
        for class_clip in class_clips:
            content = template
            pos_ed = pos[0]
            while True:
                pos_ed += 1
                if content[pos_ed] == "}":
                    break
            content = content[:pos[0]] + class_clip + content[pos_ed+1:]

            global total_data_count
            with open("data/" + prompt[:-3] + "_" + str(total_data_count[prompt]) + ".txt", 'w') as handle:
                handle.write(content)
            total_data_count[prompt] += 1

def gen_data(data_type):
    dfs(raw_data_path + data_type + "/")
    file_names = list(files.keys())
    for (prompt, template) in prompts.items():
        # All {content} {content%x} replacement
        # TODO: {prev_generated_QA} in question.md (all hybrid prompt templates)
        replace_content(file_names, prompt, template, "{content", ".md")

        # All {code} replacement
        replace_content(file_names, prompt, template, "{code}", ".py")

        # {code_class}
        extract_code_class(file_names, prompt, template)


if __name__ == "__main__":
    os.makedirs("data/", exist_ok=True)

    file_list = os.listdir(prompt_path)
    for file in file_list:
        if not file.endswith(".md"):
            continue
        with open(prompt_path + file, mode = "r") as handle:
            prompts[file] = handle.read()

    for data_type in ["IFS_code", "IFS_document"]:
        files = {}
        gen_data(data_type)
