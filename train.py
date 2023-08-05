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

from utils import get_exp_id, get_spm_dataset

from experiment_args import ScriptArguments

from accelerate import Accelerator

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
        cache_dir=script_args.cache_dir)
    
    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
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

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,
                                              trust_remote_code=True,
                                              cache_dir=script_args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


exp_id = get_exp_id(script_args.ckpt_path)

training_arguments = TrainingArguments(
    output_dir=script_args.ckpt_path + exp_id + "/",    # dummy path
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
    ddp_find_unused_parameters=False)

if script_args.only_finetune:
    procedure = ["finetune"]
else:
    procedure = ["baseline" if script_args.baseline else "pretrain", "finetune"]

# iterate multiple training stages. Usually 1 - 2 stages.
for phase in procedure:
    model, peft_config, tokenizer = create_and_prepare_model(script_args)
    model.config.use_cache = False

    training_arguments.output_dir = (script_args.ckpt_path + exp_id + "_" +
                                     phase + "/")

    # Saving the arguments for reference in the future
    os.makedirs(training_arguments.output_dir, exist_ok=True)
    script_args.dump(os.path.join(training_arguments.output_dir, "setting.yml"))

    dataset = get_spm_dataset(phase=phase,
                              mode="train",
                              with_self_instruct=script_args.with_self_instruct)

    trainer = SFTTrainer(model=model,
                         train_dataset=dataset,
                         peft_config=peft_config,
                         dataset_text_field="text",
                         max_seq_length=script_args.max_seq_length,
                         tokenizer=tokenizer,
                         args=training_arguments,
                         packing=script_args.packing)

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
    script_args.load_dir = latest_checkpoint = max(glob.glob(
        os.path.join(training_arguments.output_dir, "checkpoint-*")),
                            key=os.path.getctime)
    del model, trainer