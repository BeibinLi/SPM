import os

from datasets import load_dataset
from transformers import HfArgumentParser, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from auto_explore_copilot import RESPONSE_TEMPLATE
from experiment_args import ScriptArguments
from model_utils import create_and_prepare_model
from utils import get_exp_id, load_script_args

parser = HfArgumentParser(ScriptArguments)
script_args = load_script_args(parser.parse_args_into_dataclasses()[0])

script_args.save_steps = min(script_args.save_steps, script_args.max_steps)

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
    ddp_find_unused_parameters=False,
    gradient_checkpointing=script_args.gradient_checkpointing,
)

# Saving the arguments for reference in the future
os.makedirs(training_arguments.output_dir, exist_ok=True)
script_args.dump(os.path.join(training_arguments.output_dir, "setting.yml"))

tokenizer, peft_config, model = create_and_prepare_model(script_args)
model.config.use_cache = False

suffix = "_easy" if script_args.easy else ""

dataset = load_dataset(
    "json",
    data_files=f"data/auto_explore_dataset_markov{suffix}.jsonl",
    split="train").shuffle(seed=42)

collator = DataCollatorForCompletionOnlyLM(
    #response_template=[5103, 29901, 13]
    #if "llama" in script_args.model_name.lower() else RESPONSE_TEMPLATE,
    response_template=RESPONSE_TEMPLATE,
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
)

trainer.train()

del model, trainer
