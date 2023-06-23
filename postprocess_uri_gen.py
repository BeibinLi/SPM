"""Extract

Human-Question: question
Keywords: x, y, z
URI: uri
"""
import os, glob, re, pdb, random, json
import hashlib
import pickle
from curious_agent import CuriousAgent

FILES = glob.glob("../cscp_data/gen/uri/*.pickle")

TRAIN_OUT_FILE = "../cscp_data/gen/uri_train.jsonl"
TEST_OUT_FILE = "../cscp_data/gen/uri_test.jsonl"

hash_to_int = lambda s: int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (
    10**8)


def spawn_chat_for_uri(msgs, n=10):

    # Get URI from the system message
    assert msgs[0][0] == "system"
    system_msg = msgs[0][1]

    uri = re.findall(r"URI to the search is: (\w+)", system_msg)[0]

    # Get information from Bot's response
    questions = []
    keywords = []
    for ppl, msg in msgs:
        if ppl != "assistant":
            continue

        questions.extend(re.findall(r"QUESTION.*: (.+)", msg))
        keywords.extend(re.findall(r"KEYWORDS.*: (.+)", msg))

    n = min(n, len(questions))
    print("n:", n)

    # Using hash for seed so that we have the same randomness for the same URI
    random.seed(hash_to_int(uri))
    random.shuffle(questions)
    random.shuffle(keywords)

    return [
        f"### Human: {q}### Assistant: {uri}\nKeywords: {k}"
        for q, k in zip(questions[:n], keywords[:n])
    ]


def dump_chat(chats: list, filename: str):
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps({"text": c}) for c in chats]))


if __name__ == "__main__":
    train_data = []
    test_data = []

    for file in FILES:
        agent = CuriousAgent(api=None, system_msg="")
        agent.load(file)

        if random.random() < 0.7:
            train_data.extend(spawn_chat_for_uri(agent.msgs, 10))
        else:
            test_data.extend(spawn_chat_for_uri(agent.msgs, 10))

    dump_chat(train_data, TRAIN_OUT_FILE)
    dump_chat(test_data, TEST_OUT_FILE)