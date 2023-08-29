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

import os
import glob
import torch
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from peft.tuners.lora import LoraLayer
from trl import SFTTrainer
from termcolor import colored

from experiment_args import ScriptArguments

from accelerate import Accelerator

from llama import Llama
from model_utils import GPT_msgs_to_Llama_dialog, Llama_chat_completion

accelerator = Accelerator()
local_rank = accelerator.process_index

# Distributed
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def create_and_prepare_model(args):
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
        del base_model
    else:
        #model = PeftModel(model=base_model, peft_config=peft_config)
        model = base_model

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              trust_remote_code=True,
                                              cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer

messages = [
    {
        "role": "System",
        "content": "Hey!",
    },
    {
        "role": "User",
        "content": "Hi, how are you?",
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

dialog = GPT_msgs_to_Llama_dialog(messages)

# model = Llama.build(ckpt_dir="/home/t-rzhou/llama/7B-chat/",
#                     tokenizer_path="/home/t-rzhou/llama/tokenizer.model",
#                     max_seq_len=8192//2,
#                     max_batch_size=1,
#                     model_parallel_size=1)

# res = model.chat_completion(
#     dialogs = [dialog],
#     logprobs = True,
#     max_gen_len = 2000,
# )

# print(res[0]["generation"])




model, peft_config, tokenizer = create_and_prepare_model(script_args)

print(Llama_chat_completion(model, tokenizer, [dialog], logprobs=True, max_gen_len=2000)[0]["generation"])
