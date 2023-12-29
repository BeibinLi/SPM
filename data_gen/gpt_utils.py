import copy
import hashlib
import os
import pdb
import re

import autogen
import tiktoken
from autogen import AssistantAgent, OpenAIWrapper, UserProxyAgent, oai
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from termcolor import colored

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


CONFIG_LIST = autogen.config_list_from_json("OAI_CONFIG_LIST")

ALL_CONFIG_LIST = CONFIG_LIST

creative_reply = lambda p: oai.Completion.extract_text(
    oai.Completion.create(config_list=CONFIG_LIST, prompt=p, temperature=1))[0]

LM_CONFIG = {
    "seed": 1,
    "config_list": CONFIG_LIST,
    "temperature": 0,
    "top_p": 1
}

# reply = lambda p, cl: oai.Completion.extract_text(
#     oai.Completion.create(config_list=cl, prompt=p, temperature=0))[0]


def reply(prompt, config_list=CONFIG_LIST, model_name=None):
    if model_name and type(model_name) is str:
        config_list = autogen.oai.openai_utils.filter_config(
            config_list, {"model": [model_name]})

    assert len(config_list), "No OpenAI config found in AutoGen."
    return OpenAIWrapper.extract_text_or_function_call(
        OpenAIWrapper(config_list=config_list).create(
                              prompt=prompt,
                              temperature=0))[0]


encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_tokens = lambda msg: len(encoder.encode(msg))

if __name__ == "__main__":
    rst = reply("How are you?", model_name="gpt-3.5-turbo")
    print(rst)
    pdb.set_trace()