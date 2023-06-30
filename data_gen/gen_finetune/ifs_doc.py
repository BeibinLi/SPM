import os, sys, re, random, json
import glob
sys.path.append(".")
from data_gen.paths import *
from curious_agent import CuriousAgent

os.makedirs(finetune_data_path, exist_ok=True)

FILES = {
    "[DOC] ": glob.glob(chatlog_output_path + "question*.pickle"),
    "[IFS] ": glob.glob(chatlog_output_path + "code_quiz*.pickle")
}

TRAIN_OUT_FILE = finetune_data_path + "ifs_doc_train.jsonl"
TEST_OUT_FILE = finetune_data_path + "ifs_doc_test.jsonl"

char_set = set("0123456789:([{ ")
num_set = set("0123456789")

def gen_data(msgs):
    # Get information from Bot's response
    questions, answers = [], []
    for ppl, msg in msgs:
        if ppl != "assistant":
            continue

        while True:
            p = msg.find("QUESTION") + len("QUESTION")
            q = msg[p:].find("ANSWER") + p
            
            while msg[p] in char_set:
                p += 1
            
            question = msg[p:q].strip()

            q += len("ANSWER")
            while msg[q] in char_set:
                q += 1
            
            t = msg[q:].find("QUESTION")
            if t == -1:
                t = len(msg[q:])
            t += q

            answer = msg[q:t].strip()

            questions.append(question)
            answers.append(answer)

            if t == len(msg):
                break
            msg = msg[t:]
    
    return questions, answers


def dump_data(data: list, filename: str):
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps({"text": d}) for d in data]))


if __name__ == "__main__":
    questions, answers = [], []
    for type, files in FILES.items():
        for file in files:
            agent = CuriousAgent(api=None, system_msg="")
            agent.load(file)

            _questions, _answers = gen_data(agent.msgs)
            _questions = [type + q for q in _questions]
            questions = questions + _questions
            answers = answers + _answers

    # Split train and test: make sure a URI in the training set will never
    # appear in the testing set
    random.seed(1)
    train_set_idx = random.sample(range(len(questions)), int(len(questions) * 0.7))

    train_data, test_data = [], []
    for i in range(len(questions)):
        _data = "### Human: " + questions[i] + "\n### Assistant: " + answers[i]

        if i in train_set_idx:
            train_data.append(_data)
        else:
            test_data.append(_data)

    dump_data(train_data, TRAIN_OUT_FILE)
    dump_data(test_data, TEST_OUT_FILE)
