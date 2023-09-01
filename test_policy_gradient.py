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

import torch
from transformers import (HfArgumentParser, GenerationConfig)
import types

from experiment_args import ScriptArguments
from model_utils import (GPT_msgs_to_Llama_dialog, Llama_chat_completion,
                         calc_probs_log_probs, create_and_prepare_model)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

m1 = [
    {
        "role": "System",
        "content": "You are a code writing assistant.",
    },
    {
        "role": "User",
        "content": "Hi, how are you?",
    },
]
m2 = [
    {
        "role": "System",
        "content": "Hey!",
    },
    {
        "role": "user",
        "content": "Good to see you, LLAMA!",
    },
]

d1 = GPT_msgs_to_Llama_dialog(m1)
d2 = GPT_msgs_to_Llama_dialog(m2)

# model = Llama.build(ckpt_dir="/home/t-rzhou/llama/7B-chat/",
#                     tokenizer_path="/home/t-rzhou/llama/tokenizer.model",
#                     max_seq_len=8192//2,
#                     max_batch_size=1,
#                     model_parallel_size=1)

# res = model.chat_completion(
#     dialogs = [d1],
#     logprobs = True,
#     max_gen_len = 2000,
# )

# print(res)

model, peft_config, tokenizer = create_and_prepare_model(script_args)
# Add our customized calculation function to the model
model.calc_probs_log_probs = types.MethodType(calc_probs_log_probs, model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

# Use multinomial sampling to generate the next token:
# Set do_sample = True, num_beams = 1, and passing temperature and top_p
generation_config = GenerationConfig(
    max_length=2048,
    do_sample=True,
    num_beams=1,
    temperature=0.6,
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
#stopping_criteria = MaxLengthCriteria(generation_config.max_length)

for epoch in range(1000):
    dialogs = [d1.copy()]

    n = 3
    res = []
    for _ in range(n):
        res.append(
            Llama_chat_completion(model, tokenizer, dialogs, generation_config))

        for i in range(len(dialogs)):
            dialogs[i].append(res[-1][i]["generation"])
            dialogs[i].append({
                "role": "user",
                "content": "Got it. Can you write another code?"
            })
    print(dialogs)
    # calculate the policy gradient in the reversed order to avoid space explosion
    # assume only one dialog each time
    tot_return = 0
    for _ in range(n - 1, -1, -1):
        # set the cost to be the number of generated tokens
        tot_return += sum(1 for x in res[_][0]["generated_mask"] if x)

        input_tokens = torch.zeros((len(res[_]), res[_][0]["tokens"].shape[0]),
                                   dtype=torch.long,
                                   device=model.device)
        for i, r in enumerate(res[_]):
            input_tokens[i] = r["tokens"]

        generated_mask = [r["generated_mask"] for r in res[_]]
        generated_mask = torch.tensor(generated_mask,
                                      dtype=torch.bool,
                                      device=model.device)

        probs_log_probs = model.calc_probs_log_probs(input_tokens,
                                                     generated_mask,
                                                     generation_config,
                                                     calc_probs=False,
                                                     calc_log_probs=True)

        log_probs = probs_log_probs["log_probs"][0]
        # in this demo, we use batch of 1, so len(probs_log_probs["X"]) == 1
        # we separate each round of chat, so len(log_probs) == 1

        for i in range(len(log_probs) - 1, -1, -1):
            optimizer.zero_grad()
            loss = tot_return * log_probs[i]
            loss.backward()
            optimizer.step()

    scheduler.step()
    print(tot_return)
