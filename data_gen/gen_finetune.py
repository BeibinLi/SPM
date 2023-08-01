import os, sys, re, random, json
import glob
sys.path.append(".")
from data_gen.paths import chatlog_output_path, finetune_data_path, raw_data_path
from curious_agent import CuriousAgent

os.makedirs(finetune_data_path, exist_ok=True)

FILES = {
    "[DOC] ": glob.glob(chatlog_output_path + "question*.pickle"),
    "[IFS] ": glob.glob(chatlog_output_path + "code_quiz*.pickle"),
    "[URI] ": glob.glob(chatlog_output_path + "uri*.pickle")
}

TRAIN_OUT_FILE = finetune_data_path + "train.jsonl"
TEST_OUT_FILE = finetune_data_path + "test.jsonl"

char_set = set("0123456789:([{ ")
num_set = set("0123456789")

def gen_data_uri(type, msgs):

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

    return {
        f"### Human: {q}\n### Assistant: {type} {uri}\n": uri
        for q in questions
    }

def gen_data(type, msgs):
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
    
    return {
        f"### Human: {questions[i]}\n### Assistant: {type} {answers[i]}\n": questions[i]
        for i in range(len(questions))
    }


def dump(data: list, filename: str):
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps({"text": d}) for d in data]))


if __name__ == "__main__":
    questions, answers = [], []
    data = {}
    for type, files in FILES.items():
        for file in files:
            agent = CuriousAgent(api=None, system_msg="")
            agent.load(file)
            if type == "[URI] ":
                _data = gen_data_uri(type, agent.msgs)
            else:
                _data = gen_data(type, agent.msgs)
            
            data.update(_data)
    
    with open(raw_data_path + "ifs_train.jsonl", "r") as f:
        raw_data = [json.loads(line) for line in f]

    for (i, d) in enumerate(raw_data):
        t = d["text"]
        t = t.replace("[IFS]", "").replace("### Assistant:", "### Assistant: [IFS]")
        data.update({t: i})

    all_values = set(list(data.values()))

    random.seed(1)
    train_set = random.sample(list(all_values), int(len(all_values) * 0.7))

    train_data = [
        k for k, v in data.items() if v in train_set
    ]

    test_data = [
        k for k, v in data.items() if v not in train_set
    ]

    dump(train_data, TRAIN_OUT_FILE)
    dump(test_data, TEST_OUT_FILE)
