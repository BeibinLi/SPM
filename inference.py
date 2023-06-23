from dataclasses import dataclass, field
from typing import Optional

import torch, pdb
from datasets import load_dataset
from peft import LoraConfig
from peft import PeftModel, PeftConfig

import transformers
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
from trl.trainer import ConstantLengthDataset
from termcolor import colored

################ Constants/Variables ################
peft_model_loc = "results/checkpoint-50/"
# peft_model_id = "dfurman/falcon-40b-chat-oasst1"
cache_dir = "/mnt/data/falcon/"

model_name = "tiiuae/falcon-7b"  # public model name
device_id = 1

device_map = {"": device_id}

prompt = """### Human: What is the  PDU Amperage for A100 in Gen 7.1?
### Assistant:"""

#############

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=bnb_config,
                                             device_map=device_map,
                                             trust_remote_code=True,
                                             cache_dir=cache_dir)

config = PeftConfig.from_pretrained(peft_model_loc)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_loc)

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          trust_remote_code=True,
                                          cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token

batch = tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')
batch = batch.to(f'cuda:{device_id}')

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
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print("-" * 30)
print(colored(generated_text, "green"))

# Inspect message response in the outputs
ans = generated_text.split("### Human: ")[1].split("### Assistant: ")[-1]
print(colored(ans, "yellow"))

# pdb.set_trace()