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
                         calc_prob_log_prob)

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
d2 = GPT_msgs_to_Llama_dialog(m2)
model, peft_config, tokenizer = create_and_prepare_model(script_args)
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

generated_mask = [[] for _ in range(len(dialogs))]
for _ in range(3):
    res = Llama_chat_completion(model, tokenizer, dialogs, generation_config,
                                generated_mask)
    generated_mask = [r["generated_mask"] for r in res]
    if _ < 2:
        for i in range(len(dialogs)):
            dialogs[i].append(res[i]["generation"])
            dialogs[i].append({
                "role": "user",
                "content": "Got it. Can you write another code?"
            })

input_tokens = torch.zeros((len(res), res[0]["tokens"].shape[0]),
                           dtype=torch.long,
                           device=model.device)
for i, r in enumerate(res):
    input_tokens[i] = r["tokens"]
generated_mask = torch.tensor(generated_mask,
                              dtype=torch.bool,
                              device=model.device)

model.calc_prob_log_prob = types.MethodType(calc_prob_log_prob, model)
print(
    model.calc_prob_log_prob(input_tokens,
                             generated_mask,
                             generation_config,
                             calc_prob=True,
                             calc_log_prob=False))
