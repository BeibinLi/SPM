import json
import os
import pickle
import tiktoken

from gpt_api import get_llm
from utils import (colored_string, display_files_recursively)

import argparse
from auto_explore_sandbox import AutoExploreSandbox, extract_commands
from termcolor import colored


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
    parser.add_argument("--application_root",
                        type=str,
                        default="../Coffee_Roasting_Dataset/data/",
                        help="The folder location of the application.")
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=32768 // 2,
        help="The maximum token length in a chat. If exceed this amount, the "
        " chat will be reset.")
    parser.add_argument("--model",
                        type=str,
                        default="gpt-4",
                        help="The model to use.")
    parser.add_argument("--file_save_path",
                        type=str,
                        default="../new_and_changed_files/",
                        help="The path to save the new or changed files.")
    return parser.parse_args()


class AutoExploreCopilot():

    def __init__(self, root, temperature, top_p, max_token_length, model,
                 file_save_path):
        self.root = os.path.abspath(root).replace('\\', '/')
        self.root_dir_name = self.root.replace(os.path.basename(self.root), '')
        self.temperature = temperature
        self.top_p = top_p
        self.max_token_length = max_token_length
        self.model = model
        self.file_save_path = file_save_path
        self.api = get_llm()
        self.msgs = []    # TODO: handle multi-round user interactions.

    def answer(self, question):
        # 1. Setup memory and chat
        start_prompt = open(
            "data_gen/prompt_templates/explore_prompt_simple.md", "r").read()
        start_prompt = start_prompt.format(all_files=display_files_recursively(
            self.root),
                                           TASK=question)

        self.msgs = [("system", start_prompt), ("user", "Lets start!")]
        self.flush_msgs()

        # 2. Create sandbox environment
        self.sandbox = AutoExploreSandbox(self.root)

        # 3. Act
        self.act()

        # 4. Cleanup sandbox and environment
        del self.sandbox

        # TODO: find the final answer
        # Cache the stdout, or log, or interpretation.
        return "\n".join(self.file_save_path)

    def flush_msgs(self):
        self.msgs = [(agent, msg) for agent, msg in self.msgs if msg]
        if not hasattr(self, "last_flushed_msg"):
            self.last_flushed_msg = 0
            self.encoder = tiktoken.encoding_for_model("gpt-4")
            self.token_length = 0

        for msg in self.msgs[self.last_flushed_msg:]:
            print(colored_string(msg))
            self.token_length += len(self.encoder.encode(msg[1]))

        self.last_flushed_msg = len(self.msgs)

        # TODO: handle memory issues: e.g., cut, summarize, etc.
        # cat_warning_ = "Warning: You can only read one file at a time. " + cmd[
        # 1] + " is ignored."

        # Reset here to incorporate the last assistant messages
        if self.token_length > self.max_token_length:
            self.dump()
            self.__init__(self.root, self.temperature, self.top_p,
                          self.max_token_length, self.model, self.data_path)
            self.msgs.append(
                ("user",
                 "You have reached the maximum token length. Now restarted."))
            return

    def act(self):
        msgs_with_short_mem = self.msgs[:-1]

        response = self.api.reply(agent_name=self.msgs[-1][0],
                                  msg=self.msgs[-1][1],
                                  num_response=1,
                                  temperature=self.temperature,
                                  top_p=self.top_p,
                                  prev_msgs=msgs_with_short_mem,
                                  model=self.model)[0]

        print(colored(response, "red"))

        self.msgs.append(("assistant", response))

        self.updated_short_mem = False

        # if "[SOLUTION]" in response:
        #     self.flush_msgs()

        #     # The agent replies with a solution, inject and run it
        #     result = self.sandbox.inject_and_run(response)
        #     # pdb.set_trace()
        #     if result["stdout"] != "":
        #         self.msgs.append(("user", "Stdout: " + result["stdout"]))

        #     if result["stderr"] != "":
        #         self.msgs.append(("user", result["stderr"]))
        #     else:
        #         # Success! save the result
        #         os.makedirs(self.file_save_path, exist_ok=True)
        #         for file_name, content in result["changed_files"].items():
        #             os.makedirs(self.file_save_path +
        #                         os.path.dirname(file_name),
        #                         exist_ok=True)
        #             with open(self.file_save_path + file_name, "wb") as f:
        #                 f.write(content)
        #     self.msgs.append(("user", result["information"]))

        commands = extract_commands(response)
        for cmd in commands:
            command_output = self.sandbox.run_command(cmd)
            if cmd[0] == "exit":
                # Success! save the result
                os.makedirs(self.file_save_path, exist_ok=True)
                for file_name, content in self.sandbox.changed_files.items():
                    os.makedirs(self.file_save_path +
                                os.path.dirname(file_name),
                                exist_ok=True)
                    with open(self.file_save_path + file_name, "wb") as f:
                        f.write(content)
                # return # end of act
            self.msgs.append(("user", command_output))

        if commands == []:
            self.msgs.append(
                ("user", "Warning: You didn't give me any command. "
                 "Further explore the code repo by sending me system commands: "
                 "ls, cd, cat."))

        self.flush_msgs()

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
        root=args.application_root,
        temperature=args.temperature,
        top_p=args.top_p,
        max_token_length=args.max_token_length,
        model=args.model,
        file_save_path=os.path.abspath(args.file_save_path) + "/")
    agent.answer(
        "Plot the bean price of Excelsa between Jun 2021 and 2022 Aug.")
