from data_gen.paths import (self_instruct_data_path,
                            self_instruct_raw_data_path)
import json
from utils import save_data

TRAIN_OUT_FILE = self_instruct_data_path + "train.jsonl"
TEST_OUT_FILE = self_instruct_data_path + "test.jsonl"

if __name__ == "__main__":
    with open(self_instruct_raw_data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]
    data = {}
    for (i, d) in enumerate(raw_data):
        prompt, completion = d["prompt"], d["completion"]
        completion = completion.replace("<|endoftext|>", "")
        data.update({f"### Human: {prompt}\n### Assistant: {completion}\n": i})

    save_data(data, TRAIN_OUT_FILE, TEST_OUT_FILE)
