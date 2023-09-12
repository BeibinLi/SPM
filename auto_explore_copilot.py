import json
import os
import pickle
import tiktoken
from peft import PeftModel

from gpt_api import get_llm
from utils import (colored_string, display_files_recursively, extract_commands,
                   SUPPORTED_CMDS)
from model_utils import GPT_msgs_to_Llama_dialog, Llama_chat_completion

import argparse
from auto_explore_sandbox import AutoExploreSandbox
from transformers import AutoTokenizer, GenerationConfig
from functions.cost import AutoExploreCostFunction
from functions.terminate import (AutoExploreTerminateCriteria, AnytimeTerminate)


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
    parser.add_argument("--application_root",
                        type=str,
                        default="/home/t-rzhou/Coffee_Roasting_Dataset/data/",
                        help="The folder location of the application.")
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=2048,
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

    def __init__(
            self,
            root: str,
            temperature: float,
            top_p: float,
            max_token_length: int,
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
            need_output_msgs: bool = True):
        """
        A copilot to help language models explore a repo.

        Args:
        - `root` (str): The root directory of the repo.
        - `temperature` (float): The temperature of the language model.
        - `top_p` (float): The top_p of the language model.
        - `max_token_length` (int): The maximum total token length for chat completion.
        - `file_save_path` (str): The path to save the new or changed files.
        - `password` (str): The password to use for the sandbox.
        - `interaction_type` (str): The type of the interaction, with choices
        in ['train', 'inference', 'debug', 'inference_rl'].
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
        - `need_output_msgs` (bool): Whether to output the messages after each act.
        """
        # TODO: support terminate criteria for inference
        assert interaction_type in [
            "train", "inference", "debug", "inference_rl"
        ], ("Only support interaction type in ['train', 'inference', 'debug', "
            "'inference_rl'].")
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
        self.file_save_path = os.path.abspath(file_save_path).replace('\\', '/')

        self.root_dir_name = self.root.replace(os.path.basename(self.root), '')

        self.temperature = temperature
        self.top_p = top_p
        self.max_token_length = max_token_length

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
        self.need_output_msgs = need_output_msgs

    def answer(self, question: str, ans_cmds: list = []):
        """
        Answer a question about the repo by autonomous exploration.

        Args:
        - `question` (str): The question to answer.
        - `ans_cmds` (list): The commands as answer. Only used when debug.
        """
        self.question = question

        # 1. Setup memory and chat
        if self.interaction_type in ["train", "debug", "inference_rl"]:
            start_prompt = open(
                "data_gen/prompt_templates/auto_explore/explore_prompt_rl.md",
                "r").read()
        elif self.interaction_type == "inference":
            start_prompt = open(
                "data_gen/prompt_templates/auto_explore/explore_prompt.md",
                "r").read()

        start_prompt = start_prompt.format(all_files=display_files_recursively(
            self.root),
                                           TASK=question)
        self.msgs = [("system", start_prompt), ("user", "Lets start!")]
        # flush the messages
        self.flush_msgs()

        # store the generation logs for training
        self.generation_logs = []

        # 2. Create sandbox environment
        if self.interaction_type == "train":
            self.supported_cmds = [
                "cd", "ls", "cat", "head", "tail", "id", "exit"
            ]
        else:
            self.supported_cmds = SUPPORTED_CMDS
        self.sandbox = AutoExploreSandbox(dataset_path=self.root,
                                          password=self.password,
                                          supported_cmds=self.supported_cmds)

        # 3. Act
        if self.interaction_type == "debug":
            # Directly use inner function _act()
            for cmd in ans_cmds:
                self._act(f"```bash\n{cmd}\n```")
        else:
            self.act()

        if not self.terminate_criteria.can_terminate():
            self.generation_logs[-1]["cost"] += 1000

        # 4. Cleanup sandbox and environment
        del self.sandbox

        # TODO: find the final answer

    def flush_msgs(self):
        self.msgs = [(agent, msg) for agent, msg in self.msgs if msg]
        if not hasattr(self, "last_flushed_msg"):
            self.last_flushed_msg = 0
            self.encoder = self.tokenizer if hasattr(
                self, "tokenizer") else tiktoken.encoding_for_model("gpt-4")
            self.token_length = 0

        for msg in self.msgs[self.last_flushed_msg:]:
            if self.need_output_msgs:
                print(colored_string(msg))
            self.token_length += len(self.encoder.encode(msg[1]))

        self.last_flushed_msg = len(self.msgs)

        # TODO: handle memory issues: e.g., cut, summarize, etc.

        # Reset here to incorporate the last assistant messages
        if self.token_length > self.max_token_length:
            # self.dump()
            # self.__init__(self.root, self.temperature, self.top_p,
            #               self.max_token_length, self.model, self.data_path)
            # self.msgs.append(
            #     ("user",
            #      "You have reached the maximum token length. Now restarted."))

            raise Exception("Token limit exceeded.")

    def get_msgs(self) -> list:
        """
        Get the message history.

        Returns:
        - list: The message history.
        """

        return self.msgs

    def act(self):
        """
        Wrapper function to interact with the language model for one step
        and call the possible next act().
        """

        try:
            ret = self._act()
        except Exception as e:
            self.msgs.append(("user", str(e)))
            ret = "Continue"

        try:
            self.flush_msgs()
        except Exception:
            return

        if ret == "Continue":
            self.act()

    def _act(self, response: str = None) -> str:
        """
        Args:
        - `response` (str): The response to use for debugging.
        """
        if self.model_type == "local":
            # Get response from local model
            # Use multinomial sampling to generate the next token:
            # Set do_sample = True, num_beams = 1
            generation_config = GenerationConfig(
                max_length=self.max_token_length,
                do_sample=True,
                num_beams=1,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            ret = Llama_chat_completion(
                model=self.model,
                tokenizer=self.tokenizer,
                dialogs=[GPT_msgs_to_Llama_dialog(self.msgs)],
                generation_config=generation_config)[0]

            response = ret["generation"]["content"]
            self.generation_logs.append({
                "tokens": ret["tokens"],
                "generated_mask": ret["generated_mask"],
                "cost": 0
            })
        elif self.model_type == "remote":
            # Get response from remote model
            response = self.api.reply(agent_name=self.msgs[-1][0],
                                      msg=self.msgs[-1][1],
                                      num_response=1,
                                      temperature=self.temperature,
                                      top_p=self.top_p,
                                      prev_msgs=self.msgs[:-1],
                                      model=self.model_name)[0]
        else:
            # Use provided response
            assert response is not None, ("Must provide a response when "
                                          "debugging.")

        self.msgs.append(("assistant", response))

        commands = extract_commands(response)

        user_response_start = len(self.msgs)

        for cmd in commands:
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
                        os.makedirs(self.file_save_path, exist_ok=True)
                        for file_name, content in self.sandbox.get_changed_files(
                        ).items():
                            os.makedirs(self.file_save_path +
                                        os.path.dirname(file_name),
                                        exist_ok=True)
                            with open(self.file_save_path + file_name,
                                      "wb") as f:
                                f.write(content)
                        return "Exit"
                    else:
                        self.msgs.append(
                            ("user",
                             "Error: The terminate criteria is not met." +
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
                user_msgs=self.msgs[user_response_start:])

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
        - list: the generation logs.
        """
        return self.generation_logs


if __name__ == "__main__":
    args = get_args()
    agent = AutoExploreCopilot(
        root=args.application_root,
        temperature=args.temperature,
        top_p=args.top_p,
        max_token_length=args.max_token_length,
        file_save_path=os.path.abspath(args.file_save_path) + "/",
        password="zrl",
        interaction_type="inference_rl",
        model_type="remote",
        model_name=args.model)
    agent.answer(
    #"Plot the bean price of Excelsa between Jun 2021 and 2022 Aug."
    #"Plot employee salary by country in a map."
    #"Who is the proprietor of the cafe in Shanghai?"
    #"What is the culture statement of Opti Coffee?"
    #"Tell me details of Employee Appreciation Events."
        "What does Opti Coffee's achievement prioritize?")
