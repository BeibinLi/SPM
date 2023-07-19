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

def extract_bash_commands(response):
    commands = []
    positions = find_all_substr(response, "```bash")
    for pos in reversed(positions):
        st = pos + 7
        p = response[st:].find("```") + st
        commands.append(response[st:p].strip())
    return reversed(commands)

def parse_echo(command):
    for i in range(len(command)):
        if command[i].strip() == ">":
            return 'echo "' + "".join(command[1:i]) + '" ' + " ".join(command[i:])
    return 'echo "' + "".join(command[1]) + '" '
