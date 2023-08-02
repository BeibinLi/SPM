import os, sys, re, random, json
import glob
sys.path.append(".")
from data_gen.paths import raw_data_path, pretrain_raw_data_path
from data_gen.gen_prompts_chatlogs.general import dfs
import tiktoken

slicing_gap = 800
slicing_len = 1000

encoder = tiktoken.encoding_for_model("gpt-4")

os.makedirs(pretrain_raw_data_path, exist_ok=True)

TRAIN_OUT_FILE = pretrain_raw_data_path + "train.jsonl"
TEST_OUT_FILE = pretrain_raw_data_path + "test.jsonl"

def slice_text(str):
    ret = []
    encoded = encoder.encode(str)
    i = 0
    while i < len(encoded):
        ret.append(encoder.decode(encoded[i:i+slicing_len]))
        i += slicing_gap
        
    return ret

data = {}

def dfs(path):
    global data
    file_list = os.listdir(path)
    for file in file_list:
        if file[0] == ".":
            continue
        new_path = path + file
        if os.path.isdir(new_path):
            dfs(new_path + "/")
        else:
            try:
                with open(new_path, mode = "r") as handle:
                    content = handle.read()
                    if content.replace(" ", "").replace("\n", "") != "": # remove empty files
                        s = slice_text(content)
                        data.update({"### Assistant: " + t: new_path for t in s})
            except:
                pass

def dump(data: list, filename: str):
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps({"text": d}) for d in data]))


if __name__ == "__main__":
    for type in ["IFS_code/", "IFS_document/"]:
        dfs(raw_data_path + type)

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
