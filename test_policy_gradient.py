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
from peft import LoraConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          GenerationConfig)
from termcolor import colored
import types
from accelerate import Accelerator

from experiment_args import ScriptArguments
from model_utils import (GPT_msgs_to_Llama_dialog, Llama_chat_completion,
                         calc_probs_log_probs)

accelerator = Accelerator()
local_rank = accelerator.process_index

# Distributed
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def create_and_prepare_model(
        args: ScriptArguments) -> (PeftModel, LoraConfig, AutoTokenizer):
    """
    Create and prepare model for PEFT training.

    Args:
    - args: ScriptArguments

    Returns:
    - model: PeftModel
    - peft_config: peft config
    - tokenizer: tokenizer associated with the model
    """
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, you can accelerate training with "
                "the argument --bf16")
            print("=" * 80)

    device_map = {"": local_rank}

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=args.cache_dir)

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if args.load_dir:
        print(colored("Loading from " + args.load_dir, "green"))
        model = PeftModel.from_pretrained(model=base_model,
                                          model_id=args.load_dir,
                                          is_trainable=True,
                                          config=peft_config)
    else:
        model = PeftModel(model=base_model, peft_config=peft_config)
    del base_model

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              trust_remote_code=True,
                                              cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


m1 = [
    {
        "role": "System",
        "content": "You are a code writing assistant.",
    },
    {
        "role": "User",
        "content": "Hi, how are you?",
    },
    {
        "role": "assistant",
        "content": "Good to see you, human!",
    },
    {
        "role": "user",
        "content": "Help me with coding.",
    },
    {
        "role": "assistant",
        "content": "Sure!",
    },
    {
        "role": "user",
        "content": "Write a program to check if a number is prime or not.",
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
    {
        "role": "assistant",
        "content": "Good to see you, human!",
    },
    {
        "role": "user",
        "content": "Help me with coding.",
    },
    {
        "role": "assistant",
        "content": "Sure!",
    },
    {
        "role": "user",
        "content": "Write a program to check if a number is prime or not.",
    },
    {
        "role": "user",
        "content": "Remember to use the Sieve of Eratosthenes algorithm.",
    },
]

d1 = GPT_msgs_to_Llama_dialog(m1)

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

d2 = GPT_msgs_to_Llama_dialog(m2)
model, peft_config, tokenizer = create_and_prepare_model(script_args)
# Add our customized calculation function to the model
model.calc_probs_log_probs = types.MethodType(calc_probs_log_probs, model)
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
dialogs = [d1]

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

for _ in range(n):
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
                                                 calc_probs=True,
                                                 calc_log_probs=True)

    probs = probs_log_probs["probs"][0]

    for i in range(len(probs)):
        print(probs[i])

        model.zero_grad()
        probs[i].backward(retain_graph=True)

        tot = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'{name}:', torch.norm(param.grad))
                tot += param.numel()

        print("Total number of parameters in the gradient: ", tot)

print(dialogs)
