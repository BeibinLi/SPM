import json
from transformers import (HfArgumentParser, AutoTokenizer)
from experiment_args import ScriptArguments

from auto_explore_copilot import AutoExploreCopilot
from functions.terminate import AnytimeTerminate


def dump(data: list, filename: str):
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(d) for d in data]))


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

dataset = json.load(
    open("/home/vectorzhou/Coffee_Roasting_Dataset/file_search_coffee.json",
         "r"))

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,
                                          trust_remote_code=True,
                                          cache_dir=script_args.cache_dir)
tokenizer.pad_token = tokenizer.eos_token

auto_explore_dataset, auto_explore_dataset_easy = [], []

for data in dataset:
    #for cmds in [data["commands"], data["optimal_path"]]:
    for cmds in [data["optimal_path"]]:
        cmds = [cmd for cmd in cmds if cmd != "ls"]
        cmds.append(cmds[-1].replace("cat", "id"))
        cmds.append("exit")

        copilot = AutoExploreCopilot(
            root="/home/vectorzhou/Coffee_Roasting_Dataset/data/",
            temperature=0.6,
            top_p=0.9,
            max_token_length=32768,
            max_new_tokens=32768,
            file_save_path="new_and_changed_files/",
            interaction_type="debug",
            model_type="null",
            model=None,
            tokenizer=None,
            cost_function=None,
            terminate_criteria=AnytimeTerminate(),
            need_output_msgs=False)

        for _ in range(10):
            copilot.answer(question=data["question"], ans_cmds=cmds)

            whole_msgs = copilot.get_whole_msgs()

            auto_explore_dataset += [{
                "text": "\n".join([msg[1] for msg in msgs[:-1]]) + msgs[-1][1],
            } for msgs in whole_msgs]

            copilot.answer(question=f"Find {data['filename']}", ans_cmds=cmds)

            whole_msgs = copilot.get_whole_msgs()

            auto_explore_dataset_easy += [{
                "text": "\n".join([msg[1] for msg in msgs[:-1]]) + msgs[-1][1],
            } for msgs in whole_msgs]

dump(auto_explore_dataset, "data/auto_explore_dataset_markov.jsonl")
dump(auto_explore_dataset_easy, "data/auto_explore_dataset_markov_easy.jsonl")
