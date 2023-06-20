"""GPT-3 Client Adapter using the Substrate API.

We use adapter here instead of connecting to Substrate API directly, because
it is easier to debug. The modularization also helps to switch to other APIs 
or even local GPU models in the future.
"""
import argparse
import atexit
import copy
import hashlib
import json
import os
import pdb
import pickle
import re
import shutil
import socket
import time
import traceback
from enum import Enum

import diskcache
import numpy as np
import openai
import requests
from msal import PublicClientApplication, SerializableTokenCache
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from termcolor import colored

try:
    import flaml
    from flaml import oai as foai
    USE_FLAML = True
except Exception as e:
    print("Unable to import flaml because:", e)
    USE_FLAML = False

ps = PorterStemmer()


def persistent_hash(obj):
    # Convert the input object to a string representation
    obj_str = repr(obj).encode('utf-8')
    # Calculate the SHA-256 hash of the string representation
    hash_obj = hashlib.sha256(obj_str).hexdigest()
    # Return the hash as a string
    return hash_obj


def stem_and_hash(sentence):
    # remove all non-alphanumeric characters
    sentence = re.sub(r'[^a-zA-Z0-9]', ' ', copy.deepcopy(sentence))

    # all whitespaces to one space
    sentence = re.sub(r"\s+", " ", sentence)

    sentence = sentence.lower().split()

    stemmed = [ps.stem(s) for s in sentence if len(s)]

    stemmed = "".join(stemmed)

    return persistent_hash(stemmed)


def cache_llm_infer_result(func):
    cache_data = diskcache.Cache(".diskcache")  # setup the cache database

    def wrapper(*args, **kwargs):

        # remove the "self" or "cls" in the args
        key = f"{func.__name__}_{stem_and_hash(str(args[1:]) + str(kwargs))}"

        storage = cache_data.get(key, None)
        if storage:
            return storage["result"]

        # Otherwise, call the function and save its result to the cache file
        result = func(*args, **kwargs)

        storage = {"result": result, "args": args[1:], "kwargs": kwargs}
        cache_data.set(key, storage)

        return result

    return wrapper


def handle_prev_message_history(agent_name, msg, prev_msgs):
    if "system" in [_agent_name for _agent_name, _message in prev_msgs]:
        messages = []
    else:
        # Handle missing system message for ChatCompletion
        messages = [
            {
                "role":
                    "system",
                "content":
                    "You are an AI assistant that writes Python code to answer questions."
            },
        ]

    if prev_msgs and len(prev_msgs):
        messages += [{
            "role": _agent_name,
            "content": _message
        } for _agent_name, _message in prev_msgs]

    return messages + [{"role": agent_name, "content": msg}]


class FLAMLWrapper():

    def __init__(self, api_config_list=None):
        self.api_config_list = api_config_list  # list of dict

        print("OpenAI Key:", openai.api_key)

    @cache_llm_infer_result
    def reply(self,
              agent_name,
              msg,
              num_response=1,
              stop=None,
              model="gpt-4",
              prev_msgs=None,
              temperature=0,
              top_p=1):
        if num_response > 1:
            assert temperature > 0 or top_p < 1
        messages = handle_prev_message_history(agent_name, msg, prev_msgs)

        for i in range(len(self.api_config_list)):
            if model in ["gpt-35-turbo", "gpt-3.5-turbo"]:
                if self.api_config_list[i]["api_type"] == "azure":
                    model_norm = "gpt-35-turbo"
                else:
                    model_norm = "gpt-3.5-turbo"
            else:
                model_norm = model
            self.api_config_list[i]["model"] = model_norm

        try:
            response = foai.Completion.create(
                config_list=self.api_config_list,  # the FLAML argument
                messages=messages,
                temperature=temperature,
                n=num_response,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
        except openai.error.RateLimitError as e:
            print(e)
            time_to_sleep = re.findall(r"retry after (\d+) second", str(e))[0]
            print(
                colored(
                    f"Rate limit exceeded. Waiting for {time_to_sleep} seconds...",
                    "yellow"))
            time.sleep(int(time_to_sleep))
            return self.reply(msg,
                              num_response,
                              stop,
                              model=model,
                              prev_msgs=prev_msgs)
        except Exception as e:
            raise e

        print(response)

        return [
            response["choices"][i]["message"]["content"]
            for i in range(len(response["choices"]))
        ]


class AzureGPTClient():

    def __init__(self):

        print("OpenAI Key:", openai.api_key)

    @cache_llm_infer_result
    def reply(self,
              agent_name,
              msg,
              num_response=1,
              stop=None,
              model="gpt-4",
              prev_msgs=None,
              temperature=0,
              top_p=1):
        while True:  # Run until succeed
            try:
                return self._try_reply(agent_name=agent_name,
                                       msg=msg,
                                       num_response=num_response,
                                       stop=stop,
                                       model=model,
                                       prev_msgs=prev_msgs,
                                       temperature=temperature,
                                       top_p=top_p)
            except openai.error.RateLimitError as e:
                print(e)
                time_to_sleep = re.findall(r"retry after (\d+) second",
                                           str(e))[0]
                print(
                    colored(
                        f"(Azure) Rate limit exceeded. Waiting for {time_to_sleep} seconds...",
                        "yellow"))
                time.sleep(int(time_to_sleep))

    def _try_reply(self,
                   agent_name,
                   msg,
                   num_response=1,
                   stop=None,
                   model="gpt-4",
                   prev_msgs=None,
                   temperature=0,
                   top_p=1):
        if num_response > 1:
            assert temperature > 0 or top_p < 1

        messages = handle_prev_message_history(agent_name, msg, prev_msgs)

        if "gpt-4" in model:
            response = openai.ChatCompletion.create(engine=model,
                                                    messages=messages,
                                                    temperature=temperature,
                                                    top_p=top_p,
                                                    stop=None,
                                                    n=num_response)
            answers = [
                response["choices"][i]["message"]["content"]
                for i in range(len(response["choices"]))
            ]
        else:
            response = openai.Completion.create(engine=model,
                                                prompt=msg,
                                                temperature=temperature,
                                                max_tokens=1000,
                                                top_p=top_p,
                                                stop=stop,
                                                n=num_response)

            answers = [
                response["choices"][i]["text"]
                for i in range(len(response["choices"]))
            ]

        print(colored("response:", "green"), response)
        print(colored("ans:", "green"), answers)

        # pdb.set_trace()
        return answers


def cache_func_call(func):
    cache_data = diskcache.Cache(".diskcache")  # setup the cache database

    def wrapper(*args, **kwargs):
        key = f"{func.__name__}_{stem_and_hash(str(args) + str(kwargs))}"

        storage = cache_data.get(key, None)
        if storage:
            return storage["result"]

        # Otherwise, call the function and save its result to the cache file
        result = func(*args, **kwargs)

        storage = {"result": result, "args": args, "kwargs": kwargs}
        cache_data.set(key, storage)

        return result

    return wrapper


@cache_func_call
def get_embedding(text, model="text-embedding-ada-002"):
    openai.api_type = "azure"
    openai.api_base = "https://msrcore.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = os.getenv("CORE_AZURE_KEY").strip().rstrip()

    text = text.replace("\n", " ")
    try:
        return openai.Embedding.create(input=[text],
                                       engine=model)['data'][0]['embedding']
    except openai.error.RateLimitError as e:
        print(e)
        time_to_sleep = re.findall(r"retry after (\d+) second", str(e))[0]
        print(
            colored(
                f"(get_embedding) Rate limit exceeded. Waiting for {time_to_sleep} seconds...",
                "yellow"))
        time.sleep(int(time_to_sleep))

        return get_embedding(text, model="text-embedding-ada-002")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def is_question_inbound(question, known_questions, threshold=0.8):

    v_q = get_embedding(question)
    for x in known_questions:
        sim = cosine_similarity(v_q, get_embedding(x))
        if sim > threshold:
            return True
    return False


# %%
def get_llm() -> object:
    openai.api_type = "azure"
    openai.api_base = "https://msrcore.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = os.getenv("CORE_AZURE_KEY").strip().rstrip()
    api = AzureGPTClient()

    return api


if __name__ == "__main__":
    api = get_llm()

    print(
        api.reply("user",
                  "Give me something? For instance: ",
                  num_response=10,
                  temperature=0.1,
                  top_p=0.3,
                  prev_msgs=[("system", "You are a bot to cook Chinese food.")],
                  model="gpt-4"))
