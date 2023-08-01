import sys
import random
sys.path.append(".")
from data_gen.paths import self_instruct_data_path, self_instruct_raw_data_path
import json

TRAIN_OUT_FILE = self_instruct_data_path + "train.jsonl"
TEST_OUT_FILE = self_instruct_data_path + "test.jsonl"

def dump(data: list, filename: str):
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps({"text": d}) for d in data]))

if __name__ == "__main__":
    with open(self_instruct_raw_data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]
    data = {}
    for (i, d) in enumerate(raw_data):
        prompt, completion = d["prompt"], d["completion"]
        completion = completion.replace("<|endoftext|>", "")
        data.update({f"### Human: {prompt}\n### Assistant: {completion}\n": i})
    
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
