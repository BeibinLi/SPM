# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import torch
import types
import random
import os
from tqdm import tqdm
from transformers import (HfArgumentParser, GenerationConfig)

from utils import get_exp_id
from experiment_args import ScriptArguments
from model_utils import (calc_probs_log_probs, create_and_prepare_model,
                         get_bash_only_generated_masks)
from auto_explore_copilot import AutoExploreCopilot

from functions.cost import (NumTokenCost, KeywordCost, SynthesizedCost)
from functions.terminate import IdentifyFileTerminate
from functions.training import policy_gradient_update

root = "/home/t-rzhou/Coffee_Roasting_Dataset/data/"

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

exp_id = get_exp_id(script_args.ckpt_path)
ckpt_path = script_args.ckpt_path + exp_id + "/"
os.makedirs(ckpt_path, exist_ok=True)

tokenizer, peft_config, model = create_and_prepare_model(script_args)
# Add our customized calculation function to the model
model.calc_probs_log_probs = types.MethodType(calc_probs_log_probs, model)

optimizer = torch.optim.Adam(model.parameters(), lr=script_args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

#stopping_criteria = MaxLengthCriteria(generation_config.max_length)

dataset = json.load(open("data/file_search_coffee.json", "r"))

temperature = 0.6
top_p = 0.9

# build the cost function
num_token_cost = NumTokenCost(tokenizer)
keyword_cost = KeywordCost(keywords=["Error", "Warning"], costs=[100, 20])
synthesized_cost = SynthesizedCost(
    cost_functions=[num_token_cost, keyword_cost], weights=[1, 1])

for epoch in tqdm(range(script_args.max_steps)):
    # random sample a data
    data = random.choice(dataset)

    generation_config = GenerationConfig(
        max_length=script_args.max_seq_length,
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
                                 file_save_path="new_and_changed_files/",
                                 password="zrl",
                                 interaction_type="train",
                                 model_type="local",
                                 model=model,
                                 tokenizer=tokenizer,
                                 cost_function=synthesized_cost,
                                 terminate_criteria=IdentifyFileTerminate(
                                     data["filename"]))

    # rollout a trajectory
    copilot.answer(data["question"])

    # dump the messages
    with open(ckpt_path + "epoch_" + str(epoch + 1) + ".json", "w") as f:
        json.dump(copilot.get_msgs(), f)

    logs = copilot.get_generation_logs()
    # calculate probs and log probs for only the bash commands
    masks = get_bash_only_generated_masks(logs=logs, tokenizer=tokenizer)
    for i in range(len(logs)):
        logs[i]["generated_mask"] = masks[i]

    # update the model
    policy_gradient_update(model=model,
                           generation_config=generation_config,
                           generation_results=[logs],
                           optimizer=optimizer,
                           scheduler=scheduler)

    if (epoch + 1) % script_args.save_steps == 0:
        _ckpt_path = ckpt_path + "epoch_" + str(epoch + 1) + "/"
        os.makedirs(_ckpt_path, exist_ok=True)
        model.save_pretrained(save_directory=_ckpt_path)
        tokenizer.save_pretrained(save_directory=_ckpt_path)
