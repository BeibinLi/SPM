import argparse
import os
import random
import string

from peft import PeftModel
from transformers import AutoTokenizer, GenerationConfig

from auto_explore_sandbox import AutoExploreSandbox, LeaveoutOption
from functions.cost import AutoExploreCostFunction
from functions.terminate import AnytimeTerminate, AutoExploreTerminateCriteria
from gpt_api import get_llm
from model_utils import transformer_text_completion
from utils import (SUPPORTED_CMDS, colored_string, extract_commands,
                   list_all_actions, wrap_path)

DEBUG_MSG = True
CHOICES = string.digits + string.ascii_letters
RESPONSE_TEMPLATE = " # Response:\n"


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
        default=1024,
        help="The maximum token length in a chat. If exceed this amount, the "
        " chat will be reset.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1,
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
            repo_root: str,
            sandbox_dir: str,
            temperature: float,
            top_p: float,
            max_token_length: int,
            max_new_tokens: int,
            file_save_path: str,
            interaction_type: str,
            model_type: str,
            model_name: str = None,
            model: PeftModel = None,
            tokenizer: AutoTokenizer = None,
            cost_function: AutoExploreCostFunction = None,
            terminate_criteria: AutoExploreTerminateCriteria = AnytimeTerminate(
            ),
            leaveout_prob: float = 0,
            shuffle_action: bool = False,
            easy: bool = False,
            need_output_msgs: bool = True):
        """
        A copilot to help language models explore a repo.

        Args:
        - `repo_root` (str): The root directory of the repo.
        - `sandbox_dir` (str): The directory to store the sandbox.
        - `temperature` (float): The temperature of the language model.
        - `top_p` (float): The top_p of the language model.
        - `max_token_length` (int): The maximum total token length for chat completion.
        - `max_new_tokens` (int): The maximum new tokens.
        - `file_save_path` (str): The path to save the new or changed files.
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
        - `leaveout_prob` (float): The probability of leaving out unrelated
        files. Only used when `interaction_type` is 'train', and passed to the
        sandbox.
        - `shuffle_action` (bool): Whether to shuffle the actions.
        - `easy` (bool): Whether to use easy mode, which omits cat operations.
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
        self.repo_root = os.path.abspath(repo_root).replace('\\', '/')
        self.sandbox_dir = os.path.abspath(sandbox_dir).replace('\\', '/')
        self.file_save_path = os.path.abspath(
            os.path.join(self.sandbox_dir, file_save_path)).replace('\\', '/')

        self.temperature = temperature
        self.top_p = top_p
        self.max_token_length = max_token_length
        self.max_new_tokens = max_new_tokens

        self.interaction_type = interaction_type
        self.model_type = model_type
        if model_type == "local":
            self.model = model
            self.tokenizer = tokenizer
            if interaction_type == "train":
                self.cost_function = cost_function

            self.generation_config = GenerationConfig(
                max_length=max_token_length,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_beams=1,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elif model_type == "remote":
            self.model_name = model_name
            self.api = get_llm()

        self.terminate_criteria = terminate_criteria
        self.leaveout_prob = leaveout_prob
        self.shuffle_action = shuffle_action
        self.easy = easy
        self.need_output_msgs = need_output_msgs

    def set_question(self, question: str, target_file: str = ""):
        """
        Set the question to answer in the copilot.

        Args:
        - `question` (str): The question to answer.
        - `target_file` (str): The target file to answer the question. Only used
        when `self.interaction_type` is 'train'.
        """
        self.question = question

        self.start_prompt = open(
            "data_gen/prompt_templates/auto_explore/explore_prompt_rl_markov.md",
            "r").read()

        # Store the generation logs for training
        self.sys_infos = []
        self.generation_logs = []
        self.whole_msgs = []
        self.cmd_hisotry = []

        # Initialize the files that have been visited for command filtering
        self._catted_files = []
        self._ided_files = []

        # Create sandbox environment
        if self.interaction_type == "train":
            self.supported_cmds = [
                "cd", "ls", "cat", "head", "tail", "id", "exit"
            ]
        else:
            self.supported_cmds = SUPPORTED_CMDS
        self.sandbox = AutoExploreSandbox(
            dataset_path=self.repo_root,
            sandbox_path=self.sandbox_dir,
            supported_cmds=self.supported_cmds,
            leaveout_option=LeaveoutOption([target_file], self.leaveout_prob))

        self.is_finished = False
        self.step = 1

        ### For first step training only
        self.ans_cmd = ""

    def set_answer(self, ans_cmd: str):
        """
        For first step training only
        """
        self.ans_cmd = ans_cmd

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
        self.set_question(question=question, target_file=target_file)

        self.ans_cmds = ans_cmds.copy()

        while not self.is_finished:
            self.build_cur_msgs()
            response = self.get_response()
            self.act_with_response(response)

        self.wrap_up()

    def get_response(self) -> str:
        # Case 1: in debug mode, commands are provided
        if self.interaction_type == "debug":
            if self.ans_cmds == []:
                response = input("Input a command:")
            else:
                cmd = self.ans_cmds.pop(0)
                response = self.choices[self.cmd_list.index(cmd)]

        # Case 2: in other modes, use the language model
        if self.model_type == "local":
            # Get response from local model
            # Use multinomial sampling to generate the next token:
            # Set do_sample = True, num_beams = 1
            ret = transformer_text_completion(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=["\n".join([msg[1] for msg in self.cur_msgs])],
                generation_config=self.generation_config)[0]
            response = self.use_lm_ret(ret)

        elif self.model_type == "remote":
            # Get response from remote model
            response = self.api.reply(agent_name=self.cur_msgs[-1][0],
                                      msg=self.cur_msgs[-1][1],
                                      num_response=1,
                                      temperature=self.temperature,
                                      top_p=self.top_p,
                                      prev_msgs=self.cur_msgs[:-1],
                                      model=self.model_name)[0]

        return response

    def use_lm_ret(self, ret: dict) -> str:
        response = ret["generation"]["content"].strip(" ")
        if response == "":
            response = " "

        ret.update({"cost": 0, "step": self.step})
        self.generation_logs.append(ret)
        return response

    def wrap_up(self):
        """
        Wrap up after answering a question.
        """
        if self.interaction_type == "train":
            if self.terminate_criteria.can_terminate():
                self.generation_logs[-1]["cost"] -= 15
            else:
                self.generation_logs[-1]["cost"] += 15

        # Save the new or changed files
        os.makedirs(self.file_save_path, exist_ok=True)
        for file_name, content in self.sandbox.get_changed_files().items():
            os.makedirs(self.file_save_path + os.path.dirname(file_name),
                        exist_ok=True)
            with open(self.file_save_path + file_name, "wb") as f:
                f.write(content)

        # Cleanup sandbox and environment
        del self.sandbox

    def flush_msgs(self):
        pass

    def get_whole_msgs(self) -> list:
        """
        Get the message history.

        Returns:
        - list: The message history.
        """

        return self.whole_msgs

    def build_cur_msgs(self):
        """
        Build current messages to send to the language model.
        """
        self.cwd = os.path.relpath(self.sandbox.cwd, self.sandbox.sandbox_dir)
        files_under_cwd = os.listdir(self.sandbox.cwd)
        self.cmd_list = self._filter_commands(sandbox_cwd=self.cwd,
                                              commands=list_all_actions(
                                                  root=self.sandbox.sandbox_dir,
                                                  curr_dir=self.sandbox.cwd))

        if len(self.cmd_list) > len(CHOICES):
            print(self.repo_root)

        if self.shuffle_action:
            self.choices = random.sample(CHOICES, len(self.cmd_list))
        else:
            self.choices = CHOICES

        ### For first step training only
        if self.ans_cmd != "":
            self.ans_cmd = self.choices[self.cmd_list.index(self.ans_cmd)]

        self.cur_msgs = [
            ("system",
             self.start_prompt.format(
                 TASK=self.question,
                 CWD=self.cwd,
                 FILES_UNDER_CWD="\n".join(
                     [wrap_path(f) for f in files_under_cwd]),
                 CMD_HIST="\n".join(self.cmd_hisotry),
                 EXEC_RES="\n".join([r[1] for r in self.sys_infos]),
                 CMD_LIST="\n".join([
                     self.choices[i] + ". " + cmd
                     for i, cmd in enumerate(self.cmd_list)
                 ])) + " " + RESPONSE_TEMPLATE)
        ]
        self.sys_infos = []

        if self.need_output_msgs:
            print(colored_string(self.cur_msgs[0]))

    def act_with_response(self, response: str) -> str:
        """

        """
        try:
            ret = self._act_with_response(response)
        except Exception as e:
            ret = "Continue"
            self.sys_infos.append(("system", f"Runtime Error: {e}"))

        if self.interaction_type == "train":
            self.generation_logs[-1]["cost"] = self.cost_function.call(
                user_msgs=self.cur_msgs + self.sys_infos)

        ### For first step training only
        if self.ans_cmd != "" and response == self.ans_cmd:
            self.generation_logs[-1]["cost"] = -115

        if ret == "Exit" or self.step == 15:
            self.is_finished = True

        self.step += 1

    def _act_with_response(self, response: str) -> str:
        self.cur_msgs.append(("assistant", response))
        self.whole_msgs.append(self.cur_msgs)

        # Only consider the first command
        if response[0] in self.choices:
            idx = self.choices.index(response[0])
            if idx >= len(self.cmd_list):
                self.sys_infos.append(("user", "Error: Invalid choice."))
                return "Continue"
            commands = extract_commands(f"```bash\n{self.cmd_list[idx]}\n```",
                                        only_first=True)
        else:
            commands = []

        for cmd in commands:
            self.sys_infos.append(("user", "Executing: " + " ".join(cmd)))
            self._update_cmd_history(self.cwd, " ".join(cmd))

            if cmd[0] == "exit":
                if len(commands) > 1:
                    self.sys_infos.append((
                        "user", "Error: There are other commands. "
                        "You could only use exit standalone in a single response."
                    ))
                else:
                    try:
                        self.flush_msgs()
                    except Exception:
                        return "Exit"

                    return "Exit"
            else:
                command_output, status = self.sandbox.run_command(cmd)
                self.terminate_criteria.update_status(**status)

                if self.easy:
                    if cmd[0] == "cat":
                        command_output = ""

                self.sys_infos.append(("user", command_output))

        if commands == []:
            self.sys_infos.append(
                ("user", "Warning: You didn't give me any command. "
                 "Further explore the repo by sending me system commands: "
                 f"{', '.join(self.supported_cmds)}."))

        return "Continue"

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
    copilot = AutoExploreCopilot(
        repo_root=args.application_root,
        temperature=args.temperature,
        top_p=args.top_p,
        max_token_length=args.max_token_length,
        max_new_tokens=args.max_new_tokens,
        file_save_path=os.path.abspath(args.file_save_path) + "/",
        interaction_type="inference",
        model_type="remote",
        model_name=args.model)
    copilot.answer(
    #"Plot the price of bean Excelsa between Jun 2021 and 2022 Aug."
    #"Plot employee salary by country in a map."
    #"Who is the proprietor of the cafe in Shanghai?"
    #"What is the culture statement of Opti Coffee?"
    #"Tell me details of Employee Appreciation Events."
        "What does Opti Coffee's achievement prioritize?")

    logs = copilot.get_generation_logs()
    print(logs)
