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
from transformers import (HfArgumentParser)
import types
import random

from experiment_args import ScriptArguments
from model_utils import (calc_probs_log_probs, create_and_prepare_model)
from auto_explore_copilot import AutoExploreCopilot

from training_funcs import (NumTokenCost, ReachFileTerminate)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

tokenizer, peft_config, model = create_and_prepare_model(script_args)
# Add our customized calculation function to the model
model.calc_probs_log_probs = types.MethodType(calc_probs_log_probs, model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

#stopping_criteria = MaxLengthCriteria(generation_config.max_length)

root = "/home/t-rzhou/Coffee_Roasting_Dataset/data/"

dataset = json.load(open("data/search_coffee.json", "r"))

for epoch in range(script_args.max_steps):
    # random sample a data
    data = random.choice(dataset)
    print(data)

    copilot = AutoExploreCopilot(root=root,
                                 temperature=0.6,
                                 top_p=0.9,
                                 max_length=2048,
                                 file_save_path="new_and_changed_files/",
                                 password="zrl",
                                 interaction_type="train",
                                 model_type="local",
                                 model=model,
                                 tokenizer=tokenizer,
                                 cost_function=NumTokenCost(tokenizer),
                                 terminate_criteria=ReachFileTerminate(
                                     data["filename"]))

    copilot.answer(data["question"])
