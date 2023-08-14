import csv
import io
import json
import os
import pickle

import tiktoken

from gpt_api import get_llm
from utils import (colored_string, find_all_substr, display_files_recursively)

import argparse
from auto_explore_dataset_wrapper import AutoExploreDatasetWrapper


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--temperature",
                        type=float,
                        default=1,
                        help="Temperature of language model.")
    parser.add_argument("--top_p",
                        type=float,
                        default=0.3,
                        help="Top_p of language model.")
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=32768 // 2,
        help="The maximum token length in a chat. If exceed this amount, the "
        " chat will be reset.")
    parser.add_argument("--model",
                        type=str,
                        default="gpt-35-turbo",
                        help="The model to use.")
    parser.add_argument("--file_save_path",
                        type=str,
                        default="new_and_changed_files/",
                        help="The path to save the new or changed files.")
    return parser.parse_args()


class AutoExploreCopilot():

    def __init__(self, root, temperature, top_p, max_token_length, model,
                 file_save_path, task, dataset_wrapper):
        self.root = os.path.abspath(root)
        self.temperature = temperature
        self.top_p = top_p
        self.max_token_length = max_token_length
        self.model = model
        self.file_save_path = file_save_path
        self.dataset_wrapper = dataset_wrapper

        self.api = get_llm()

        start_prompt = open(
            "data_gen/prompt_templates/explore_prompt_simple.md", "r").read()
        start_prompt = start_prompt.format(
            all_files=display_files_recursively(root), TASK=task)

        self.msgs = [("system", start_prompt), ("user", "Lets start!")]

        self.encoder = tiktoken.encoding_for_model("gpt-4")
        self.token_length = sum(
            [len(self.encoder.encode(msg[1])) for msg in self.msgs])

        for msg in self.msgs:
            print(colored_string(msg))

        os.chdir(root)

    def get_cwd(self):
        return os.getcwd().replace('\\', '/').replace(
            self.root.replace(os.path.basename(self.root), ''), '')

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
        if cmd[0] in ["cd", "cat"]:
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
                     (f"Error: You cannot access files ({cwd}) outside"
                      f"the repo ({self.root})! You are now at {os.getcwd()}")))
                return

        if cmd[0] not in ["cd", "ls", "cat"]:
            self.msgs.append(
                ("user", "Error: You can only run cd, ls, cat commands."))
            return

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
        except Exception as e:
            self.msgs.append(("user", "Error: " + str(e)))

    def act(self):
        self.read_count = 0

        msgs_with_short_mem = self.msgs[:-1]

        response = self.api.reply(agent_name=self.msgs[-1][0],
                                  msg=self.msgs[-1][1],
                                  num_response=1,
                                  temperature=self.temperature,
                                  top_p=self.top_p,
                                  prev_msgs=msgs_with_short_mem,
                                  model=self.model)[0]

        self.msgs.append(("assistant", response))

        unencoded_pos = len(self.msgs) - 1

        self.updated_short_mem = False

        if "[SOLUTION]" in response:
            # Flush all un-printed messages
            for msg in self.msgs[unencoded_pos:]:
                print(colored_string(msg))
            unencoded_pos = len(self.msgs)

            # The agent replies with a solution, inject and run it
            result = self.dataset_wrapper.inject_and_run(response)
            if result["stderr"] != "":
                self.msgs.append(("user", result["stderr"]))
            else:
                # save the result
                os.makedirs(self.file_save_path, exist_ok=True)
                for file_name, content in result["changed_files"].items():
                    os.makedirs(self.file_save_path +
                                os.path.dirname(file_name),
                                exist_ok=True)
                    with open(self.file_save_path + file_name, "wb") as f:
                        f.write(content)

                exit(0)
        else:
            commands = self.extract_commands(response)
            for cmd in commands:
                self.handle_command(cmd)

        if commands == []:
            self.msgs.append(
                ("user", "Warning: You didn't give me any command. "
                 "Further explore the code repo by sending me system commands: "
                 "ls, cd, cat."))

        # Reset here to incorporate the last assistant messages
        if self.token_length > self.max_token_length:
            self.dump()
            self.__init__(self.root, self.temperature, self.top_p,
                          self.max_token_length, self.model, self.data_path)
            self.msgs.append(
                ("user",
                 "You have reached the maximum token length. Now restarted."))
            self.token_length += len(self.encoder.encode(self.msgs[-1][1]))
            return

        self.token_length += sum([
            len(self.encoder.encode(msg[1]))
            for msg in self.msgs[unencoded_pos:]
        ])

        for msg in self.msgs[unencoded_pos:]:
            print(colored_string(msg))

        agent.act()

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
    agent = AutoExploreCopilot(
        root="../Coffee_Roasting_Dataset",
        temperature=args.temperature,
        top_p=args.top_p,
        max_token_length=args.max_token_length,
        model=args.model,
        file_save_path=os.path.abspath(args.file_save_path) + "/",
        task="Plot the bean price of Excelsa between Jun 2021 and 2022 Aug.",
        dataset_wrapper=AutoExploreDatasetWrapper("../Coffee_Roasting_Dataset"))
    agent.act()
