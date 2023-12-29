import json
import os

from transformers import AutoTokenizer, GenerationConfig, HfArgumentParser

from auto_explore_copilot import AutoExploreCopilot
from experiment_args import ScriptArguments
from functions.terminate import AnytimeTerminate
from utils import load_dataset, unwrap_path


def dump(data: list, filename: str):
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(d) for d in data]))


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,
                                          trust_remote_code=True,
                                          cache_dir=script_args.cache_dir)
tokenizer.pad_token = tokenizer.eos_token

auto_explore_dataset, auto_explore_dataset_easy = [], []

dataset = load_dataset(script_args.task_file)

generation_config = GenerationConfig(
    max_length=script_args.max_seq_length,
    max_new_tokens=script_args.max_new_tokens,
    do_sample=True,
    num_beams=1,
    temperature=script_args.temperature,
    top_p=script_args.top_p,
    top_k=script_args.top_k,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

for data in dataset:
    if "root" not in data.keys():
        # Only for file_search_coffee.json
        root = "coffee_roasting_dataset"
    else:
        root = data["root"]
    root = os.path.join(script_args.repo_dir, root)

    #for cmds in [data["commands"], data["optimal_path"]]:
    for _cmds in [data["optimal_path"]]:
        cmds = []
        for cmd in _cmds:
            if cmd == "ls":
                continue
            for op in ["cd", "cat"]:
                if cmd.startswith(op):
                    file = unwrap_path(cmd.replace(op, "").strip())
                    cmd = op + " " + file
                    break
            cmds.append(cmd)

        cmds.append(cmds[-1].replace("cat", "id"))
        cmds.append("exit")

        copilot = AutoExploreCopilot(repo_root=root,
                                     sandbox_dir=script_args.sandbox_dir,
                                     horizon=15,
                                     generation_config=generation_config,
                                     file_save_path="new_and_changed_files/",
                                     interaction_type="debug",
                                     model_type="null",
                                     model=None,
                                     tokenizer=None,
                                     cost_function=None,
                                     terminate_criteria=AnytimeTerminate(),
                                     leaveout_prob=0,
                                     shuffle_action=True,
                                     need_output_msgs=False)

        copilot.easy_mode = False
        copilot.answer(question=data["question"],
                       target_file=data["filename"],
                       ans_cmds=cmds)

        whole_msgs = copilot.get_whole_msgs()

        auto_explore_dataset += [{
            "text": "\n".join([msg[1] for msg in msgs[:-1]]) + msgs[-1][1],
        } for msgs in whole_msgs]

        copilot.easy = True
        copilot.answer(question=f"Find {data['filename']}",
                       target_file=data["filename"],
                       ans_cmds=cmds)

        whole_msgs = copilot.get_whole_msgs()

        auto_explore_dataset_easy += [{
            "text": "\n".join([msg[1] for msg in msgs[:-1]]) + msgs[-1][1],
        } for msgs in whole_msgs]

dump(auto_explore_dataset, "data/auto_explore_dataset_markov.jsonl")
dump(auto_explore_dataset_easy, "data/auto_explore_dataset_markov_easy.jsonl")
