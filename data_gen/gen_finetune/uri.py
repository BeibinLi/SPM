import os, sys, re, random, json
import glob
sys.path.append(".")
from data_gen.paths import *
from curious_agent import CuriousAgent

os.makedirs(finetune_data_path, exist_ok=True)

FILES = glob.glob(chatlog_output_path + "uri*.pickle")

TRAIN_OUT_FILE = finetune_data_path + "uri_train.jsonl"
TEST_OUT_FILE = finetune_data_path + "uri_test.jsonl"

def gen_data(msgs):

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

    random.seed(0)
    random.shuffle(questions)
    random.shuffle(keywords)

    return {uri: (questions, keywords)}


def dump_data(data: list, filename: str):
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps({"text": d}) for d in data]))


if __name__ == "__main__":
    data = {}  # key: uri, value: questions

    for file in FILES:
        agent = CuriousAgent(api=None, system_msg="")
        agent.load(file)

        data.update(gen_data(agent.msgs))

    all_uris = set(list(data.keys()))

    # Split train and test: make sure a URI in the training set will never
    # appear in the testing set
    random.seed(1)
    train_set_uri = random.sample(list(all_uris), int(len(all_uris) * 0.7))

    train_data, test_data = [], []
    for uri, (questions, keywords) in data.items():
        _data = ["### Human: [URI] " + question + "\n### Assistant: " + uri for question in questions]

        if uri in train_set_uri:
            train_data = train_data + _data
        else:
            test_data = test_data + _data

    dump_data(train_data, TRAIN_OUT_FILE)
    dump_data(test_data, TEST_OUT_FILE)
