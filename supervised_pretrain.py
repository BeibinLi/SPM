import glob
import os
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft.tuners.lora import LoraLayer
from transformers import HfArgumentParser, TrainingArguments

from auto_explore_copilot import RESPONSE_TEMPLATE
from experiment_args import ScriptArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from model_utils import load_script_args, create_and_prepare_model
from utils import get_exp_id

accelerator = Accelerator()
local_rank = accelerator.process_index

# Distributed
parser = HfArgumentParser(ScriptArguments)
script_args = load_script_args(parser.parse_args_into_dataclasses()[0])

exp_id = get_exp_id(script_args.ckpt_path)

training_arguments = TrainingArguments(
    output_dir=script_args.ckpt_path + exp_id + "_supervised_pretrain/",
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

# Saving the arguments for reference in the future
os.makedirs(training_arguments.output_dir, exist_ok=True)
script_args.dump(os.path.join(training_arguments.output_dir, "setting.yml"))

tokenizer, peft_config, model = create_and_prepare_model(script_args)
model.config.use_cache = False

if "llama" in script_args.model_name.lower():
    tokenizer.padding_side = 'right'
    dataset = load_dataset(
        "json",
        data_files="data/auto_explore_dataset_markov_llama.jsonl",
        split="train").shuffle(seed=42)
else:
    dataset = load_dataset(
        "json",
        data_files="data/auto_explore_dataset_markov_gpt.jsonl",
        split="train").shuffle(seed=42)

collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE + "\n",
                                           tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    data_collator=collator,
    args=training_arguments,
    #packing=script_args.packing
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
script_args.load_dir = max(glob.glob(
    os.path.join(training_arguments.output_dir, "checkpoint-*")),
                           key=os.path.getctime)
del model, trainer
