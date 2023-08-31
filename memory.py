"""Agent memory.
This module is designed to manage memory for multi-agent interactions.

OpenAI-based solutions manages memory for 2-agent environment. With more
multi-agent interactions, the memory management becomes more complicated.
An agent has memory for itself and memory for each of its contacts.
While interacting with a contact, the agent needs to retrieve the memory
for itself and the memory for all the contacts. For instance, in the real world,
while a student is discussing coding with a friend, he/she should have access
to memory while chatting with another friend.

We use time-based indexing to manage memory because
(a) it is similar to the real world.
(b) some time-consuming function calls can be non-blocking.

Hypothesis:
1. The memory supports multi-agent interactions.
2. The system message between two agents are constant.
"""
import time
import numpy as np
import pandas as pd

import concurrent.futures
import tiktoken


# non-blocking call
def embed(text: str, model: str = "text-embedding-ada-002") -> list:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_embed, text, model)
        return future.result()


def _embed(text, model):
    # TODO: implement
    return np.zeros(shape=[
        1000,
    ])


# non-blocking call
# TODO: maybe, we need agent's memory here too.
def summary(text: str, model: str) -> str:
    return ""


class AgentMemory:

    def __init__(
        self,
        agent_name: str,
        model_name: str,
        message_token_threshold: int = 1000,
    ):
        self.agent_name = agent_name
        self.message_token_threshold = message_token_threshold
        self.model_name = model_name

        self.history = {}
        self.mental_state = []
        self.contacts_system_messages = {}
        self.encoder = tiktoken.encoding_for_model(
            model_name.replace("35", "3.5").replace("-32k", ""))

    def _add_msg(self, sender, receiver, text):
        timestamp = time.time()
        self.history[timestamp] = {}
        self.history[timestamp]["sender"] = sender
        self.history[timestamp]["receiver"] = receiver
        self.history[timestamp]["text"] = text
        self.history[timestamp]["text_embedding"] = embed(
            text)    # non-blocking call
        self.history[timestamp]["tokens"] = len(self.encoder.encode(text))
        self.history[timestamp]["summary"] = self.summary(text) if self.history[
            timestamp]["tokens"] > self.message_token_threshold else ""

    def set_sys_msg(self, contact_name, text):
        assert contact_name not in ["self", "system"]
        self._add_msg("system", contact_name, text)

    def add_memory_from(self, contact_name, text):
        assert contact_name not in ["self", "system"]
        self._add_msg(contact_name, "self", text)

    def add_memory_to(self, contact_name, text):
        assert contact_name not in ["self", "system"]
        self._add_msg("self", contact_name, text)

    def save(self, csv_name):
        df = pd.DataFrame(self.history)
        df.to_csv(csv_name)

    def load(self, csv_name):
        df = pd.read_csv(csv_name)
        self.history = dict(df)    # TODO: check

    def retrieve_memory(self,
                        contact_name,
                        text,
                        format: str = "openai",
                        memory_token_limit: int = 10000) -> list:
        # TODO: should be handled with "add_memory_from"

        # Approaches: naive cut, retrieve selected, summary blocks
        #

        return


class GlobalMemory:
    # Help book-keeping all agents' memories.
    def __init__(self):
        pass
