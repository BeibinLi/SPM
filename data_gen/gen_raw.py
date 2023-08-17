from data_gen.paths import raw_data_path, pretrain_raw_data_path
from utils import save_data, slice_text
import os

slicing_gap = 800
slicing_len = 1000

os.makedirs(pretrain_raw_data_path, exist_ok=True)

TRAIN_OUT_FILE = pretrain_raw_data_path + "train.jsonl"
TEST_OUT_FILE = pretrain_raw_data_path + "test.jsonl"

data = {}


def raw_dfs(path):
    # TODO: why not combines with data_gen.gen_prompts_chatlogs.general.dfs
    global data
    file_list = os.listdir(path)
    for file in file_list:
        if file[0] == ".":
            continue
        new_path = path + file
        if os.path.isdir(new_path):
            raw_dfs(new_path + "/")
        else:
            try:
                with open(new_path, mode="r") as handle:
                    content = handle.read()
                    if content.replace(" ", "").replace(
                            "\n", "") != "":    # remove empty files
                        s = slice_text(content)
                        data.update(
                            {"### Assistant: " + t: new_path for t in s})
            except Exception as e:
                del e


if __name__ == "__main__":
    for type in ["IFS_code/", "IFS_document/"]:
        raw_dfs(raw_data_path + type)

    save_data(data, TRAIN_OUT_FILE, TEST_OUT_FILE)
