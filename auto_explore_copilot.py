import argparse
import json
import os
import pdb
import pickle
import string

from peft import PeftModel
from transformers import AutoTokenizer, GenerationConfig

from auto_explore_sandbox import AutoExploreSandbox, LeaveoutOption
from functions.cost import AutoExploreCostFunction
from functions.terminate import AnytimeTerminate, AutoExploreTerminateCriteria
from gpt_api import get_llm
from model_utils import transformer_text_completion
from utils import (SUPPORTED_CMDS, colored_string, wrap_path, extract_commands,
                   list_all_actions)

DEBUG_MSG = True
WAIT_FOR_INPUT = "<<<Wait for input>>>"
CHOICES = string.digits + string.ascii_letters
RESPONSE_TEMPLATE = "# Your command (use only the choice)"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--temperature",
                        type=float,
                        default=0.6,
                        help="Temperature of language model.")
    parser.add_argument("--top_p",
                        type=float,
                        default=0.9,
                        help="Top_p of language model.")
    parser.add_argument(
        "--application_root",
        type=str,
        default="/home/vectorzhou/Coffee_Roasting_Dataset/data/",
        help="The folder location of the application.")
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=2048,
        help="The maximum token length in a chat. If exceed this amount, the "
        " chat will be reset.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="The maximum new tokens in a chat. If exceed this amount, the "
        " chat will be reset.")
    parser.add_argument(
        "--model",
        type=str,
    #default="gpt-35-turbo",
        default="tuned",
        help=
        "The model to use. Use Huggingface model name, or 'tuned' or 'original'."
    )
    parser.add_argument("--file_save_path",
                        type=str,
                        default="new_and_changed_files/",
                        help="The path to save the new or changed files.")
    return parser.parse_args()


class AutoExploreCopilot():

    def __init__(
            self,
            root: str,
            temperature: float,
            top_p: float,
            max_token_length: int,
            max_new_tokens: int,
            file_save_path: str,
            password: str,
            interaction_type: str,
            model_type: str,
            model_name: str = None,
            model: PeftModel = None,
            tokenizer: AutoTokenizer = None,
            cost_function: AutoExploreCostFunction = None,
            terminate_criteria: AutoExploreTerminateCriteria = AnytimeTerminate(
            ),
            leaveout_fraction: float = 0,
            need_output_msgs: bool = True):
        """
        A copilot to help language models explore a repo.

        Args:
        - `root` (str): The root directory of the repo.
        - `temperature` (float): The temperature of the language model.
        - `top_p` (float): The top_p of the language model.
        - `max_token_length` (int): The maximum total token length for chat completion.
        - `max_new_tokens` (int): The maximum new tokens.
        - `file_save_path` (str): The path to save the new or changed files.
        - `password` (str): The password to use for the sandbox.
        - `interaction_type` (str): The type of the interaction, with choices
        in ['train', 'inference', 'debug'].
        - `model_type` (str): The type of the model to use, with choices
        in ['local', 'remote', 'null']. If `interaction_type` is 'train', then
        must be 'local'.
        - `model_name` (str): The name of the model to use. Only used when
        `model_type` is 'remote'.
        - `model` (PeftModel): The model to use, only support Llama 2.
        Only used when `model_type` is 'local'.
        - `tokenizer` (AutoTokenizer): The tokenizer to use. Only used when
        `model_type` is 'local'.
        - `cost_function` (AutoExploreCostFunction): The cost function to use.
        Input is the list of messages, output is the cost. Only used when
        `interaction_type` is 'train'.
        - `terminate_criteria` (AutoExploreTerminateCriteria): The terminate
        criteria for an interaction. Input is the list of messages, output is
        True / False.
        - `leaveout_fraction` (float): The probability of leaving out unrelated
        files. Only used when `interaction_type` is 'train', and passed to the
        sandbox.
        - `need_output_msgs` (bool): Whether to output the messages after each act.
        """
        assert interaction_type in [
            "train", "inference", "debug"
        ], ("Only support interaction type in ['train', 'inference', 'debug'].")
        assert model_type in [
            "local", "remote", "null"
        ], ("Only support model ype in ['local', 'remote', 'null'].")

        if interaction_type == "train":
            assert model_type == "local", "Only support local model for training."
        if interaction_type == "inference":
            assert model_type != "null", "Must provide a model for inference."

        if model_type == "local":
            assert (model is not None
                    and tokenizer is not None), ("For local model, provide the "
                                                 "model and the tokenizer.")
            if interaction_type == "train":
                assert cost_function is not None, ("For training, provide the "
                                                   "cost function.")
        elif model_type == "remote":
            assert model_name is not None, ("For remote model, provide the "
                                            "model name.")

        # replace all paths with absolute paths
        self.root = os.path.abspath(root).replace('\\', '/')
        try:
            self.file_save_path = os.path.abspath(file_save_path).replace(
                '\\', '/')
        except Exception:
            pdb.set_trace()

        self.root_dir_name = self.root.replace(os.path.basename(self.root), '')

        self.temperature = temperature
        self.top_p = top_p
        self.max_token_length = max_token_length
        self.max_new_tokens = max_new_tokens

        self.password = password
        self.interaction_type = interaction_type
        self.model_type = model_type
        if model_type == "local":
            self.model = model
            self.tokenizer = tokenizer
            if interaction_type == "train":
                self.cost_function = cost_function
        elif model_type == "remote":
            self.model_name = model_name
            self.api = get_llm()

        self.terminate_criteria = terminate_criteria
        self.leaveout_fraction = leaveout_fraction
        self.need_output_msgs = need_output_msgs

    def answer(self, question: str, target_file: str = "", ans_cmds: list = []):
        """
        Answer a question about the repo by autonomous exploration.

        Args:
        - `question` (str): The question to answer.
        - `target_file` (str): The target file to answer the question. Only used
        when `self.interaction_type` is 'train'.
        - `ans_cmds` (list): The commands of answer, can be either optimal or
        random (but still correct). Only used when debug.
        """
        self.question = question
        if self.interaction_type != "train":
            assert target_file == "", "Only support target file for training."

        # 1. Setup memory and chat
        # if self.interaction_type in ["train", "debug"]:
        #     self.start_prompt = open(
        #         "data_gen/prompt_templates/auto_explore/explore_prompt_rl_markov.md",
        #         "r").read()
        # elif self.interaction_type == "inference":
        #     self.start_prompt = open(
        #         "data_gen/prompt_templates/auto_explore/explore_prompt.md",
        #         "r").read()
        self.start_prompt = open(
            "data_gen/prompt_templates/auto_explore/explore_prompt_rl_markov.md",
            "r").read()

        # store the generation logs for training
        self.msgs = []
        self.generation_logs = []
        self.whole_msgs = []
        self.cmd_hisotry = []

        # Initialize the files that have been visited for command filtering
        self._catted_files = []
        self._ided_files = []

        # 2. Create sandbox environment
        if self.interaction_type == "train":
            self.supported_cmds = [
                "cd", "ls", "cat", "head", "tail", "id", "exit"
            ]
        else:
            self.supported_cmds = SUPPORTED_CMDS
        self.sandbox = AutoExploreSandbox(
            dataset_path=self.root,
            password=self.password,
            supported_cmds=self.supported_cmds,
            leaveout_option=LeaveoutOption([target_file],
                                           self.leaveout_fraction))

        # 3. Act
        self.step = 0
        if self.interaction_type == "debug":
            # Directly use inner function _act()
            if ans_cmds == []:
                while True:
                    ret = self._act(WAIT_FOR_INPUT)
                    if ret == "Exit":
                        break
            else:
                for cmd in ans_cmds:
                    self._act(cmd)
        else:
            self.act()

        if not self.terminate_criteria.can_terminate():
            self.generation_logs[-1]["cost"] += 1000

        # 4. Save the new or changed files
        os.makedirs(self.file_save_path, exist_ok=True)
        for file_name, content in self.sandbox.get_changed_files().items():
            os.makedirs(self.file_save_path + os.path.dirname(file_name),
                        exist_ok=True)
            with open(self.file_save_path + file_name, "wb") as f:
                f.write(content)

        # 5. Cleanup sandbox and environment
        del self.sandbox

        # TODO: find the final answer

    def flush_msgs(self):
        pass

    def get_whole_msgs(self) -> list:
        """
        Get the message history.

        Returns:
        - list: The message history.
        """

        return self.whole_msgs

    def act(self):
        """
        Wrapper function to interact with the language model for one step
        and call the possible next act().
        """
        # try:
        #     ret = self._act()
        # except Exception as e:
        #     self.msgs.append(("user", str(e)))
        #     ret = "Continue"
        ret = self._act()

        if ret == "Continue" and self.step < 15:
            self.act()

    def _act(self, cmd: str = None) -> str:
        """
        Args:
        - `cmd` (str): The command to use for debugging.
        """
        cwd = os.path.relpath(self.sandbox.cwd, self.sandbox.sandbox_dir)
        files_under_cwd = os.listdir(self.sandbox.cwd)
        cmd_list = self._filter_commands(sandbox_cwd=cwd,
                                         commands=list_all_actions(
                                             root=self.sandbox.sandbox_dir,
                                             curr_dir=self.sandbox.cwd,
                                         ))
        cur_msgs = [
            ("system",
             self.start_prompt.format(
                 TASK=self.question,
                 CWD=cwd,
                 FILES_UNDER_CWD="\n".join(
                     [wrap_path(f) for f in files_under_cwd]),
                 CMD_HIST="\n".join(self.cmd_hisotry),
                 EXEC_RES="\n".join([msg[1] for msg in self.msgs]),
                 CMD_LIST="\n".join([
                     CHOICES[i] + ". " + cmd for i, cmd in enumerate(cmd_list)
                 ])) + RESPONSE_TEMPLATE)
        ]
        self.msgs = []

        if self.need_output_msgs:
            print(colored_string(cur_msgs[0]))

        if self.model_type == "local":
            # Get response from local model
            # Use multinomial sampling to generate the next token:
            # Set do_sample = True, num_beams = 1
            generation_config = GenerationConfig(
                max_length=self.max_token_length,
            # the `max_new_tokens` will override the `max_token_length`
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                num_beams=1,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # ret = Llama_chat_completion(
            #     model=self.model,
            #     tokenizer=self.tokenizer,
            #     dialogs=[GPT_msgs_to_Llama_dialog(cur_msgs)],
            #     generation_config=generation_config)[0]
            ret = transformer_text_completion(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=["\n".join([msg[1] for msg in cur_msgs])],
                generation_config=generation_config)[0]
            response = ret["generation"]["content"]

            ret.update({"cost": 0, "step": self.step})
            self.generation_logs.append(ret)
        elif self.model_type == "remote":
            # Get response from remote model
            response = self.api.reply(agent_name=cur_msgs[-1][0],
                                      msg=cur_msgs[-1][1],
                                      num_response=1,
                                      temperature=self.temperature,
                                      top_p=self.top_p,
                                      prev_msgs=cur_msgs[:-1],
                                      model=self.model_name)[0]
        else:
            # Use provided response
            assert cmd is not None, ("Must provide a response when "
                                     "debugging.")
            if cmd == WAIT_FOR_INPUT:
                response = input("Input a command:")
            else:
                response = CHOICES[cmd_list.index(cmd)]

        # Increment step after querying the language model
        self.step += 1

        cur_msgs.append(("assistant", response))
        self.whole_msgs.append(cur_msgs)

        if self.need_output_msgs:
            print(colored_string(cur_msgs[1]))

        # Only consider the first command
        if response[0] in CHOICES:
            idx = CHOICES.index(response[0])
            commands = extract_commands(f"```bash\n{cmd_list[idx]}\n```",
                                        only_first=True)
        else:
            commands = []

        for cmd in commands:
            self.msgs.append(("user", "Executing: " + " ".join(cmd)))
            self._update_cmd_history(cwd, " ".join(cmd))

            if cmd[0] == "exit":
                if len(commands) > 1:
                    self.msgs.append((
                        "user", "Error: There are other commands. "
                        "You could only use exit standalone in a single response."
                    ))
                else:
                    try:
                        self.flush_msgs()
                    except Exception:
                        return "Exit"

                    if self.terminate_criteria.can_terminate():
                        # Success! save the result
                        return "Exit"
                    else:
                        self.msgs.append(
                            ("user",
                             "Error: The terminate criteria is not met. " +
                             self.terminate_criteria.describe_criteria()))
                        return "Continue"
            else:
                command_output, status = self.sandbox.run_command(
                    cmd, self.password)
                self.terminate_criteria.update_status(**status)

                self.msgs.append(("user", command_output))

        if commands == []:
            self.msgs.append(
                ("user", "Warning: You didn't give me any command. "
                 "Further explore the repo by sending me system commands: "
                 f"{', '.join(self.supported_cmds)}."))

        if self.interaction_type == "train":
            self.generation_logs[-1]["cost"] = self.cost_function.call(
                user_msgs=self.msgs)

        return "Continue"

    def dump(self):
        """
        Dump the current state of the agent to a pickle file specified by
        `self.data_path`.
        """
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

    def get_generation_logs(self):
        """
        Get the generation logs for training.

        Returns:
        - list: A list of generation logs, each log following format:
        {
            "generation":
            {
                "role": str,
                "content": str,
            }
            "tokens": torch.Tensor,
            "generated_mask": list,
            "cost": float,
            "step": int
        }
        """
        return self.generation_logs

    def _filter_commands(self, sandbox_cwd: str, commands: list) -> list:
        """
        Filter out available commands based on the cwd in the sandbox and
        command history. Prevents repeated access of a same file.

        Args:
        - `sandbox_cwd` (str): The cwd in the sandbox.
        - `commands` (list): The commands to filter.

        Returns:
        - list: The available commands.
        """
        ret = []
        for command in commands:
            if command.startswith("cat"):
                file = sandbox_cwd + command[4:]
                if file not in self._catted_files:
                    ret.append(command)
            elif command.startswith("id"):
                file = sandbox_cwd + command[3:]
                if file not in self._ided_files:
                    ret.append(command)
            else:
                ret.append(command)
        return ret

    def _update_cmd_history(self, sandbox_cwd: str, command: string):
        """
        Maintain command history.

        Args:
        - `sandbox_cwd` (str): The cwd in the sandbox.
        - `command` (str): The command to execute.
        """
        self.cmd_hisotry.append(command)
        if command.startswith("cat"):
            file = sandbox_cwd + command[4:]
            self._catted_files.append(file)
        elif command.startswith("id"):
            file = sandbox_cwd + command[3:]
            self._ided_files.append(file)


if __name__ == "__main__":
    args = get_args()
    agent = AutoExploreCopilot(
        root=args.application_root,
        temperature=args.temperature,
        top_p=args.top_p,
        max_token_length=args.max_token_length,
        max_new_tokens=args.max_new_tokens,
        file_save_path=os.path.abspath(args.file_save_path) + "/",
        password="zrl",
        interaction_type="inference",
        model_type="remote",
        model_name=args.model)
    agent.answer(
    #"Plot the price of bean Excelsa between Jun 2021 and 2022 Aug."
    #"Plot employee salary by country in a map."
    #"Who is the proprietor of the cafe in Shanghai?"
    #"What is the culture statement of Opti Coffee?"
    #"Tell me details of Employee Appreciation Events."
    #"What does Opti Coffee's achievement prioritize?"
        "What is Opti Coffee's achievement?")
