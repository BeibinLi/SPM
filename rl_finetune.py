import json
import os
import random
import types

import torch
from tqdm import tqdm
from transformers import (GenerationConfig, HfArgumentParser, GPT2Tokenizer,
                          GPT2ForSequenceClassification)

from auto_explore_copilot import AutoExploreCopilot
from experiment_args import ScriptArguments
from functions.cost import StepCost, KeywordCost, NumTokenCost, SynthesizedCost
from functions.terminate import IdentifyFileTerminate
from functions.training import policy_gradient_update
from model_utils import (load_script_args, calc_probs_log_probs,
                         create_and_prepare_model)
from utils import build_curriculum, get_exp_id

root = os.path.expanduser("~/Coffee_Roasting_Dataset/data")

parser = HfArgumentParser(ScriptArguments)
script_args = load_script_args(parser.parse_args_into_dataclasses()[0])

exp_id = get_exp_id(script_args.ckpt_path)
output_dir = script_args.ckpt_path + exp_id + "_rl_finetune/"
os.makedirs(output_dir, exist_ok=True)

# Saving the arguments for reference in the future
os.makedirs(output_dir, exist_ok=True)
script_args.dump(os.path.join(output_dir, "setting.yml"))

# Setup policy network
tokenizer, peft_config, model = create_and_prepare_model(script_args)
# Add our customized calculation function to the model
model.calc_probs_log_probs = types.MethodType(calc_probs_log_probs, model)

optimizer = torch.optim.Adam(model.parameters(), lr=script_args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

# Setup value network
value_model = GPT2ForSequenceClassification.from_pretrained(
    'gpt2', num_labels=1).cuda()
value_model.config.pad_token_id = value_model.config.eos_token_id
value_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
value_tokenizer.pad_token = value_tokenizer.eos_token
value_optimizer = torch.optim.Adam(value_model.parameters(), lr=1e-2)

# Build dataset
dataset = build_curriculum(
    json.load(open(os.path.join(root, "..", "file_search_coffee.json"), "r")))

temperature = 0.6
top_p = 0.9

# build the cost function
num_token_cost = NumTokenCost(tokenizer)
keyword_cost = KeywordCost(keywords=["Error", "Warning"], costs=[100, 20])
step_cost = StepCost()
synthesized_cost = SynthesizedCost(
    cost_functions=[num_token_cost, keyword_cost], weights=[1, 1])

step_per_curriculum = script_args.max_steps // len(dataset)
script_args.max_steps = step_per_curriculum * len(dataset)

# set up first curriculum
cur_dataset_idx = -1
cur_dataset = []
losses = []

for epoch in tqdm(range(script_args.max_steps)):
    # move on to the next curriculum
    if epoch % step_per_curriculum == 0:
        cur_dataset_idx += 1
        cur_dataset += dataset[cur_dataset_idx]

    # random sample a data
    data = random.choice(cur_dataset)

    generation_config = GenerationConfig(
        max_length=script_args.max_seq_length,
        max_new_tokens=script_args.max_new_tokens,
        do_sample=True,
        num_beams=1,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # setup the copilot
    copilot = AutoExploreCopilot(root=root,
                                 temperature=temperature,
                                 top_p=top_p,
                                 max_token_length=script_args.max_seq_length,
                                 max_new_tokens=script_args.max_new_tokens,
                                 file_save_path="new_and_changed_files/",
                                 interaction_type="train",
                                 model_type="local",
                                 model=model,
                                 tokenizer=tokenizer,
                                 cost_function=step_cost,
                                 terminate_criteria=IdentifyFileTerminate(
                                     data["filename"]),
                                 leaveout_fraction=0.5,
                                 need_output_msgs=False)

    # rollout a trajectory
    copilot.answer(question=data["question"], target_file=data["filename"])

    logs = copilot.get_generation_logs()

    # calculate probs and log probs for only the bash commands
    # masks = get_bash_only_generated_masks(logs=logs, tokenizer=tokenizer)
    # for i in range(len(logs)):
    #     logs[i]["generated_mask"] = masks[i]

    # update the model
    losses.append(
        policy_gradient_update(model=model,
                               generation_config=generation_config,
                               generation_results=[logs],
                               optimizer=optimizer,
                               scheduler=scheduler,
                               value_model=value_model,
                               value_tokenizer=value_tokenizer,
                               value_optimizer=value_optimizer))

    if (epoch + 1) % script_args.logging_steps == 0:
        print(losses, sum(losses) / len(losses))
        losses = []

    if (epoch + 1) % script_args.save_steps == 0:
        ckpt_path = output_dir + "epoch_" + str(epoch + 1) + "/"
        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(save_directory=ckpt_path)
        tokenizer.save_pretrained(save_directory=ckpt_path)

        # dump the messages
        with open(ckpt_path + "msgs.json", "w") as f:
            f.write("\n".join(
                [json.dumps(line) for line in copilot.get_whole_msgs()]))
