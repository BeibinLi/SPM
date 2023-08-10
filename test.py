import csv
import io
import json
import os
import pickle

import tiktoken

from gpt_api import get_llm
from utils import (get_exp_id, colored_string, find_all_substr,
                   display_files_recursively)

import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",
                        type=str,
                        required=True,
                        help="The directory of the code repo to explore.")
    parser.add_argument("--temperature",
                        type=float,
                        default=1,
                        help="Temperature of language model.")
    parser.add_argument("--top_p",
                        type=float,
                        default=0.3,
                        help="Top_p of language model.")
    parser.parse_args()
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=32768 // 2,
        help="The maximum token length in a chat. If exceed this amount, the "
        " chat will be reset.")
    parser.add_argument("--model",
                        type=str,
                        default="gpt-4-32k",
                        help="The model to use.")

    return parser.parse_args()


class AutoExploreCopilot():

    def __init__(self, root, temperature, top_p, max_token_length, model,
                 data_path):
        os.chdir(root)

        self.root = root
        self.temperature = temperature
        self.top_p = top_p
        self.max_token_length = max_token_length
        self.model = model
        self.data_path = data_path

        self.long_mem_path = root + "/long_mem.txt"
        if not os.path.exists(self.long_mem_path):
            with open(self.long_mem_path, "w"):
                pass

        self.short_mem_path = root + "/short_mem.txt"
        try:
            with open(self.short_mem_path, "r") as f:
                self.short_mem = f.read()
        except Exception as e:
            del e
            print("Initialize short-term memory")
            self.short_mem = ""

        self.api = get_llm()

        start_prompt = open("data_gen/prompt_template.md", "r").read()
        start_prompt = start_prompt.format(
            root=os.path.basename(root),
            root2=os.path.basename(root),
            all_files=display_files_recursively(root))

        self.msgs = [("system", start_prompt), ("user", "Lets start!")]

        self.encoder = tiktoken.encoding_for_model("gpt-4")
        self.token_length = sum(
            [len(self.encoder.encode(msg[1])) for msg in self.msgs])

        for msg in self.msgs:
            print(colored_string(msg))

    def get_cwd(self):
        return os.getcwd().replace('\\', '/').replace(
            root.replace(os.path.basename(root), ''), '')

    def extract_bash_commands(self, response, identifier="```bash"):
        commands = []
        positions = find_all_substr(response, identifier)
        for pos in positions:
            st = pos + len(identifier)
            p = response[st:].find("```") + st
            commands.append(response[st:p].strip())
        return commands

    def parse_echo(self, command):
        for i in range(len(command)):
            if command[i].strip().startswith(">"):
                assert i == len(command) - 2
                return [
                    "echo", '"' + "".join(command[1:i]) + '"',
                    command[i].strip(), command[i + 1]
                ]
        return ["echo", '"' + "".join(command[1:]) + '"']

    def extract_commands(self, response):
        response = response.replace("'", '"')
        bash_commands = self.extract_bash_commands(response)

        parsed_commands = []

        for bash_command in bash_commands:
            f = io.StringIO(bash_command)
            reader = csv.reader(f,
                                delimiter=' ',
                                quotechar='"',
                                skipinitialspace=True)
            for row in reader:
                if row == []:
                    continue
                if row[0] == "echo":
                    parsed_commands.append(self.parse_echo(row))
                else:
                    parsed_commands.append(row)

        return parsed_commands

    def handle_command(self, cmd):
        # Test outside repo
        if cmd[0] in ["cd", "cat", "echo"]:
            if cmd[0] == "echo" and len(cmd) != 4:
                self.msgs.append(
                    ("user",
                     "Warning: echo command without output file, ignored."))
                return

            cmd[-1] = cmd[-1].strip()
            path = os.path.dirname(cmd[-1]) if "." in os.path.basename(
                cmd[-1]) else cmd[-1]
            if path == "":
                path = "."
            original_cwd = os.getcwd()
            try:
                os.chdir(path)
            except Exception as e:
                self.msgs.append(("user", "Error: " + str(e)))
                return
            cwd = os.getcwd().replace('\\', '/')
            os.chdir(original_cwd)
            if not os.path.abspath(cwd).startswith(os.path.abspath(self.root)):
                self.msgs.append(
                    ("user",
                     "Error: You cannot access files outside the repo!"))
                return

            if cmd[0] == "echo":
                # TODO: check if it writes to repo files
                if cmd[-1].endswith("long_mem.txt") and cwd != self.root:
                    cmd[-1] = self.root + "/long_mem.txt"
                    self.msgs.append(
                        ("user",
                         "Warning: long_mem.txt must be at the root of repo! "
                         "The file path is redirected to root."))
                if cmd[-1].endswith("short_mem.txt") and cwd != self.root:
                    cmd[-1] = self.root + "/short_mem.txt"
                    self.msgs.append(
                        ("user",
                         "Warning: short_mem.txt must be at the root of repo, "
                         "and you do not need to use echo to update it! "
                         "The file path is redirected to root."))

        try:
            if cmd[0] == "cd":
                os.chdir(cmd[1])
                self.msgs.append(("user", "Now at: " + self.get_cwd()))
            else:
                ret = os.popen(" ".join(cmd)).read()
                if cmd[0] == "ls":
                    self.msgs.append(("user", "The result of ls is:\n" + ret))
                elif cmd[0] == "cat":
                    self.read_count += 1
                    if self.read_count == 1:
                        self.msgs.append(
                            ("user",
                             "The content of " + cmd[1] + " is:\n" + ret))
                    else:
                        self.msgs.append(
                            ("user",
                             "Warning: You can only read one file at a time. " +
                             cmd[1] + " is ignored."))
                elif cmd[0] == "echo":
                    if cmd[-1].endswith("short_mem.txt"):
                        self.updated_short_mem = True
                    self.msgs.append(("user", "Echo success!"))
        except Exception as e:
            self.msgs.append(("user", "Error: " + str(e)))

    def act(self):
        self.read_count = 0

        msgs_with_short_mem = self.msgs[:-1] + [
            ("assistant",
             f'---- Current short_mem.txt file. Please update it! ----\n'
             f'{open(os.path.join(self.root, "short_mem.txt"), "r").read()}')
        ]

        response = self.api.reply(agent_name=self.msgs[-1][0],
                                  msg=self.msgs[-1][1],
                                  num_response=1,
                                  temperature=self.temperature,
                                  top_p=self.top_p,
                                  prev_msgs=msgs_with_short_mem,
                                  model=self.model)[0]

        self.msgs.append(("assistant", response))

        if self.token_length > self.max_token_length:
            self.dump()
            self.__init__(self.temperature, self.top_p, self.max_token_length,
                          self.model, self.data_path)
            self.msgs.append((
                "user",
                "You just restarted the task. You may need to read long memory "
                "to pick up the progress."))
            self.token_length += len(self.encoder.encode(self.msgs[-1][1]))
            return

        unencoded_pos = len(self.msgs) - 1

        self.updated_short_mem = False

        commands = self.extract_commands(response)
        for cmd in commands:
            self.handle_command(cmd)

        if commands == []:
            self.msgs.append((
                "user",
                "Warning: You didn't give me any command. Further explore the "
                "code repo by sending me system commands: ls, cd, cat, and echo."
            ))

        if response.find("#UpdateShortMem") != -1:
            mem_blocks = self.extract_bash_commands(response,
                                                    "```short_mem.txt")
            if mem_blocks != []:
                self.short_mem = mem_blocks[0].strip()
                with open(self.short_mem_path, "w") as f:
                    f.write(self.short_mem)
                self.updated_short_mem = True

        if self.updated_short_mem:
            self.msgs.append(("user", "Short memory updated!"))
        else:
            self.msgs.append(("user", "Warning: No update to short memory."))

        # Reset here to incorporate the last assistant messages
        if self.token_length > self.max_token_length:
            self.dump()
            self.__init__(self.root, self.temperature, self.top_p,
                          self.max_token_length, self.model, self.data_path)
            self.msgs.append(
                ("user",
                 "You have reached the maximum token length. Now restarted. "
                 "You may need to read long memory to pick up the progress."))
            self.token_length += len(self.encoder.encode(self.msgs[-1][1]))
            return

        self.token_length += sum([
            len(self.encoder.encode(msg[1]))
            for msg in self.msgs[unencoded_pos:]
        ])

        for msg in self.msgs[unencoded_pos:]:
            print(colored_string(msg))

    def dump(self):
        ckpts = os.listdir(self.data_path)
        ckpts = [x.replace(".pickle", "") for x in ckpts]
        ckpt_num_list = [int(x) for x in ckpts if x.isdigit()]
        ckpt_id = max(ckpt_num_list) + 1 if ckpt_num_list != [] else 0
        ckpt = str(ckpt_id).zfill(5)
        out_loc = os.path.join(self.data_path, ckpt + ".pickle")
        with open(out_loc, "wb") as f:
            pickle.dump([
                self.msgs, self.temperature, self.top_p, self.max_token_length,
                self.model, self.data_path
            ], f)
        with open(out_loc.replace(".pickle", ".json"), "w") as f:
            json.dump([
                self.msgs, self.temperature, self.top_p, self.max_token_length,
                self.model, self.data_path
            ],
                      f,
                      indent=4)


if __name__ == "__main__":
    args = get_args()
    root = args.dir

    if not os.path.exists(os.path.join(root, "long_mem.txt")):
        open(os.path.join(root, "long_mem.txt"), "w").write("")
    if not os.path.exists(os.path.join(root, "short_mem.txt")):
        open(os.path.join(root, "short_mem.txt"), "w").write("")

    if not os.path.exists(root):
        print("ROOT not found!")
        exit(1)

    os.makedirs(args.data_path, exist_ok=True)
    exp_id = get_exp_id(args.data_path)
    data_path = os.path.abspath(os.path.join(args.data_path, exp_id))
    os.makedirs(data_path, exist_ok=True)

    agent = AutoExploreCopilot(root=root,
                               temperature=args.temperature,
                               top_p=args.top_p,
                               max_token_length=args.max_token_length,
                               model=args.model,
                               data_path=data_path)
    while True:
        agent.act()
