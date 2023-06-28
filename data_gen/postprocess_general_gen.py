"""Extract

Task prompts: explain / rewrite / compare / summarize / correlation / logic
Desired reply:
"""
import os, sys, glob, re, random, json
import hashlib
from paths import *
sys.path.append("..")
from SPM.curious_agent import CuriousAgent

FILES = glob.glob(chatlog_output_path + "*rewrite*.pickle") + \
        glob.glob(chatlog_output_path + "*explain*.pickle") + \
        glob.glob(chatlog_output_path + "*compare*.pickle") + \
        glob.glob(chatlog_output_path + "*summarize*.pickle") + \
        glob.glob(chatlog_output_path + "*correlation*.pickle") + \
        glob.glob(chatlog_output_path + "*logic*.pickle")

TRAIN_OUT_FILE = data_path + "general_train.jsonl"
TEST_OUT_FILE = data_path + "general_test.jsonl"

hash_to_int = lambda s: int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (
    10**8)

os.makedirs(data_path, exist_ok=True)

def spawn_chat(msgs, n=10):

    # Get URI from the system message
    assert msgs[0][0] == "system"
    system_msg = msgs[0][1] # Task prompts

    responses = []

    # Get information from Bot's response
    for ppl, msg in msgs:
        if ppl != "assistant":
            continue
        responses.append(msg)
    n = min(n, len(responses))

    # Using hash for seed so that we have the same randomness for the same URI
    random.seed(hash_to_int(system_msg))
    random.shuffle(responses)

    return {system_msg: responses[:n]}

def dump_chat(chats: list, filename: str):
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps({"text": c}) for c in chats]))


if __name__ == "__main__":
    data = {}  # key: question; value: uri

    for file in FILES:
        agent = CuriousAgent(api=None, system_msg="")
        agent.load(file)

        data.update(spawn_chat(agent.msgs, 10))

    prompts = list(data.keys())
    random.seed(1)
    train_set = random.sample(prompts, int(len(prompts) * 0.7))

    train_data = [
        (prompt, response) for prompt, response in data.items() if prompt in train_set
    ]

    test_data = [
        (prompt, response) for prompt, response in data.items() if prompt not in train_set
    ]

    dump_chat(train_data, TRAIN_OUT_FILE)
    dump_chat(test_data, TEST_OUT_FILE)