# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

import os
from dataclasses import dataclass, field
from typing import Optional

import torch.distributed as dist
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from peft.tuners.lora import LoraLayer
from termcolor import colored
from torch.utils.data.distributed import DistributedSampler
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          TrainingArguments)
from trl import SFTTrainer

from config import *


def get_exp_id(ckpt_path: str):
    os.makedirs(ckpt_path, exist_ok=True)
    exp_dirs = os.listdir(ckpt_path)
    exp_num_list = [int(x) for x in exp_dirs if x.isdigit()]
    exp_id = max(exp_num_list) + 1 if exp_num_list != [] else 0
    return str(exp_id).zfill(3)


def initialize_accelerator():
    accelerator = Accelerator()
    local_rank = accelerator.process_index
    return local_rank


def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    device_map = {"": local_rank}
    bnb_config = get_bnb_config(args, compute_dtype)
    base_model = load_base_model(args, device_map, bnb_config)

    if args.load_dir:
        model = load_model_from_dir(args, base_model)
    else:
        model = base_model

    peft_config = get_peft_config()
    tokenizer = get_tokenizer(args)

    return model, peft_config, tokenizer


def get_bnb_config(args, compute_dtype):
    return BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )


def load_base_model(args, device_map, bnb_config):
    return AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )


def load_model_from_dir(args, base_model):
    print(colored("Loading from " + args.load_dir, "green"))
    model = PeftModel.from_pretrained(model=base_model, model_id=args.load_dir, is_trainable=True)
    del base_model
    return model


def get_peft_config():
    return LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
    )


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              trust_remote_code=True,
                                              cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_training_arguments(args, exp_id):
    return TrainingArguments(
        output_dir=ckpt_path + exp_id + "/",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        ddp_find_unused_parameters=False
    )


def train_model(args, model, peft_config, tokenizer, training_arguments):
    model.config.use_cache = False
    train_dataset = load_training_dataset(args)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=args.packing
    )

    set_model_properties(trainer, args)
    trainer.train(resume_from_checkpoint=args.load_dir)


def load_training_dataset(args):
    return load_dataset(
        "json",
        data_files=[
            finetune_data_path + "ifs_doc_train.jsonl",
            finetune_data_path + "uri_train.jsonl",
            finetune_data_path + "ifs_train.jsonl"
        ],
        split="train"
    ).shuffle(seed=42)


def set_model_properties(trainer, args):
    for name, module in trainer.model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    local_rank = initialize_accelerator()

    model, peft_config, tokenizer = create_and_prepare_model(script_args)
    training_arguments = get_training_arguments(script_args, get_exp_id(ckpt_path))
    train_model(script_args, model, peft_config, tokenizer, training_arguments)
    print("Done")


if __name__ == "__main__":
    main()
