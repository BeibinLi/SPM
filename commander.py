import os

from gpt_api import get_llm

from auto_explore_sandbox import AutoExploreSandbox

from abc import abstractmethod

api = get_llm()


class Agent:

    def __init__(self,
                 name: str = "",
                 temperature: float = 0,
                 top_p: float = 1,
                 max_token_length: int = 1000,
                 model_name="gpt-4"):
        self.name = name
        self.temperature = temperature
        self.top_p = top_p
        self.max_token_length = max_token_length
        self.model_name = model_name
        self.memory = {}    # key: agent name; value: list of chat history

    @abstractmethod
    def answer(self, question: str) -> str:
        raise NotImplementedError

    def organize_memory(self):
        for agent_name in self.memory:
            self.memory[agent_name].sort(key=lambda x: x[0])


class PrivateCommander(Agent):

    def __init__(self, name, root, file_save_path, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.root = os.path.abspath(root).replace('\\', '/')

        self.root_dir_name = self.root.replace(os.path.basename(self.root), '')
        self.file_save_path = file_save_path

    def add_soldier(self, agent):
        if agent.name not in self.memory:
            self.memory[agent.name] = []

    def answer_user(self, question: str) -> str:
        # 1. Setup memory and chat
        self.msgs = [("system", "Try to answer the question"),
                     ("user", question)]
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

    def interact_with_sandbox(self, action: dict):

        return
