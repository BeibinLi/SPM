import glob
import os
import pdb
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel
from peft.tuners.lora import LoraLayer
from termcolor import colored
# from transformers.models import AutoModelForCausalLM
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          TrainingArguments)
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from config import *
from accelerate import Accelerator

################ Constants/Variables ################
list_all_checkpoints = lambda: glob.glob(ckpt_path + "checkpoint-*")
# peft_model_id = "dfurman/falcon-40b-chat-oasst1"

accelerator = Accelerator()
device_map = {"": accelerator.process_index}

default_question = "What is the  PDU Amperage for A100 in Gen 7.1?"

#############


def load_latest_model():
    global config, model
    checkpoints = list_all_checkpoints()
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(colored(f"Loading model from {latest_checkpoint}", "yellow"))
    config = PeftConfig.from_pretrained(latest_checkpoint)
    model = PeftModel.from_pretrained(llm_model, latest_checkpoint)


################

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)

llm_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=bnb_config,
                                                 device_map=device_map,
                                                 trust_remote_code=True,
                                                 cache_dir=model_path)

# Load the Lora model
load_latest_model()

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          trust_remote_code=True,
                                          cache_dir=model_path)
tokenizer.pad_token = tokenizer.eos_token


def answer(question):
    _prompt = f"### Human: {question}### Assistant:"
    batch = tokenizer(_prompt,
                      padding=True,
                      truncation=True,
                      return_tensors='pt')
    batch = batch.to(f'cuda:{accelerator.process_index}')

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(
            input_ids=batch.input_ids,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(output_tokens[0],
                                      skip_special_tokens=True)

    # print(colored(generated_text, "green"))

    # Inspect message response in the outputs
    ans = generated_text.split("### Human: ")[1].split("### Assistant: ")[-1]

    return ans


if __name__ == "__main__":

    ################
    print("Example: ", default_question)
    print("Bot:", colored(answer(default_question), "green"))
    ################

    while True:
        print("-" * 30)
        question = input("Human: ")

        question = question.strip().rstrip()

        if question == "":
            continue

        if question == "quit":
            break
        elif question == "pdb":
            pdb.set_trace()
        elif question == "load":
            load_latest_model()
        else:
            ans = answer(question)
            print("Bot:", colored(ans, "green"))
