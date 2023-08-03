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
from dataclasses import dataclass, field
from typing import Optional

import yaml
import pdb
import os
import torch
from peft import LoraConfig, PeftModel
# from transformers.models import AutoModelForCausalLM
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

from utils import get_exp_id, get_spm_dataset

import argparse

from accelerate import Accelerator
accelerator = Accelerator()
local_rank = accelerator.process_index


# Distributed


########################################################################
# This is a fully working simple example to use trl's RewardTrainer.
#
# This example fine-tunes any causal language model (GPT-2, GPT-Neo, etc.)
# by using the RewardTrainer from trl, we will leverage PEFT library to finetune
# adapters on the model.
#
########################################################################


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    # local_rank: Optional[int] = field(default=-1,
    #                                   metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="model/llama2/7B-chat",
        metadata={
            "help":
                "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )    
    ckpt_path: Optional[str] = field(
        default="results/",
        metadata={
            "help":
                "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of training epochs for the reward model."
        },
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help":
                "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    max_steps: int = field(
        default=5000,
        metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=False,
        metadata={
            "help":
                "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(
        default=10, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: int = field(
        default=100, metadata={"help": "Limit the total amount of checkpoints. Deletes the older checkpoints."})
    logging_steps: int = field(default=10,
                               metadata={"help": "Log every X updates steps."})
    cache_dir: Optional[str] = field(
        default="model/",
        metadata={"help": "Where to store the pretrained models."})
    
    load_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to load the pretrained models. None for no loading. latest for latest checkpoint. directory for loading from a directory."})
    
    with_self_instruct: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use self-instruct data."})
    
    baseline: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether be in baseline mode, i.e., only pretrain on raw data."})
    
    def load(self, yaml_file: str):
        with open(yaml_file, 'r') as file:
            yaml_data = yaml.safe_load(file)

        for key, value in yaml_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

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
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )
            print("=" * 80)

    device_map = {"": local_rank}

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=script_args.cache_dir
    )
    
    if args.load_dir:
        print(colored("Loading from " + args.load_dir, "green"))
        model = PeftModel.from_pretrained(model=base_model, model_id=args.load_dir, is_trainable=True)
        del base_model
    else:
        model = base_model

    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=[
        #     "query_key_value",
        #     "dense",
        #     "dense_h_to_4h",
        #     "dense_4h_to_h",
        # ],
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,
                                              trust_remote_code=True,
                                              cache_dir=script_args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer

exp_id = get_exp_id(script_args.ckpt_path)

training_arguments = TrainingArguments(
    output_dir=script_args.ckpt_path + exp_id + "/", # dummy path
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    ddp_find_unused_parameters=False
)

model, peft_config, tokenizer = create_and_prepare_model(script_args)
model.config.use_cache = False

procedure = ["baseline" if script_args.baseline else "pretrain", "finetune"]

# iterate multiple training stages. Usually 1 - 2 stages.
for phase in procedure:
    training_arguments.output_dir = script_args.ckpt_path + exp_id + "_" + phase + "/"
    
    # Saving the arguments for reference in the future
    os.makedirs(training_arguments.output_dir, exist_ok=True)
    yaml.dump(script_args.__dict__, open(os.path.join(training_arguments.output_dir, "setting.yml"), "w"))
    pdb.set_trace()

    dataset = get_spm_dataset(phase=phase, mode="train", with_self_instruct=script_args.with_self_instruct)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=script_args.packing
    )


    for name, module in trainer.model.named_modules():
        if isinstance(module, LoraLayer):
            if script_args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if script_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    trainer.train()

if local_rank == 0:
    print(colored("="*10, "green"))
    print(colored("Done", "green"))
    print(colored("="*10, "green"))