"""Extract

Task prompts: explain / rewrite / compare / summarize / correlation / logic
Desired reply:
"""
import os, sys, glob, re, random, json
import hashlib
sys.path.append(".")
from data_gen.paths import *
from curious_agent import CuriousAgent


uri_files = glob.glob(chatlog_output_path + "uri*.pickle")
general_files = glob.glob(chatlog_output_path + "*rewrite*.pickle") + \
    glob.glob(chatlog_output_path + "*explain*.pickle") + \
    glob.glob(chatlog_output_path + "*compare*.pickle") + \
    glob.glob(chatlog_output_path + "*summarize*.pickle") + \
    glob.glob(chatlog_output_path + "*correlation*.pickle") + \
    glob.glob(chatlog_output_path + "*logic*.pickle")

TRAIN_OUT_FILE = pretrain_data_path + "train.jsonl"
TEST_OUT_FILE = pretrain_data_path + "test.jsonl"

hash_to_int = lambda s: int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (
    10**8)

os.makedirs(pretrain_data_path, exist_ok=True)

def spawn_chat_for_uri(msgs, n=10):

    # Get URI from the system message
    assert msgs[0][0] == "system"
    system_msg = msgs[0][1]

    uri = re.findall(r"URI to the search is: (.+)", system_msg)[0]

    # Get information from Bot's response
    questions = []
    keywords = []
    for ppl, msg in msgs:
        if ppl != "assistant":
            continue

        questions.extend(re.findall(r"QUESTION.*: (.+)", msg))
        keywords.extend(re.findall(r"KEYWORDS.*: (.+)", msg))

    n = min(n, len(questions))
    #print("n:", n)

    # Using hash for seed so that we have the same randomness for the same URI
    random.seed(hash_to_int(uri))
    random.shuffle(questions)
    random.shuffle(keywords)

    return {
        f"### Human: {q}\n### Assistant: Keywords: {k}\nURI: {uri}\n": uri
        for q, k in zip(questions[:n], keywords[:n])
    }

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

    return {
        f"### Human: {system_msg}\n### Assistant: {r}\n": system_msg
        for r in responses[:n]
    }

def dump_chat(chats: list, filename: str):
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps({"text": c}) for c in chats]))


if __name__ == "__main__":
    data = {}  # key: question; value: uri

    for file in uri_files:
        agent = CuriousAgent(api=None, system_msg="")
        agent.load(file)

        data.update(spawn_chat_for_uri(agent.msgs, 10))

    for file in general_files:
        agent = CuriousAgent(api=None, system_msg="")
        agent.load(file)

        data.update(spawn_chat(agent.msgs, 10))

    all_values = set(list(data.values()))

    random.seed(1)
    train_set = random.sample(list(all_values), int(len(all_values) * 0.7))

    train_data = [
        k for k, v in data.items() if v in train_set
    ]

    test_data = [
        k for k, v in data.items() if v not in train_set
    ]

    dump_chat(train_data, TRAIN_OUT_FILE)
    dump_chat(test_data, TEST_OUT_FILE)
    