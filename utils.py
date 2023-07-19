import os


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

def extract_bash_commands(response, identifier="```bash"):
    commands = []
    positions = find_all_substr(response, identifier)
    for pos in reversed(positions):
        st = pos + 7
        p = response[st:].find("```") + st
        commands.append(response[st:p].strip())
    return reversed(commands)

def parse_echo(command):
    for i in range(len(command)):
        if command[i].strip()[0] == ">":
            return 'echo "' + "".join(command[1:i]) + '" ' + " ".join(command[i:])
    return 'echo "' + "".join(command[1]) + '" '

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