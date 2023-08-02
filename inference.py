import glob
import os
import pdb

import torch
from peft import PeftConfig, PeftModel
from termcolor import colored
# from transformers.models import AutoModelForCausalLM
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from config import model_name, model_path, ckpt_path
from utils import get_spm_dataset

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None, required=True, help="Dir to load model")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "manual"], help="Mode: 'auto' for auto testing on random samples, 'manual' for manual input")
    return parser.parse_args()

def load_inference_model(dir):
    global tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=False,
    )

    llm_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    quantization_config=bnb_config,
                                                    trust_remote_code=True,
                                                    cache_dir=model_path)

    # Load the Lora model
    load_latest_model(llm_model, dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            trust_remote_code=True,
                                            cache_dir=model_path)
    tokenizer.pad_token = tokenizer.eos_token


def load_latest_model(llm_model, dir):
    global config, model
    list_all_checkpoints = lambda x: glob.glob(x + "checkpoint-*")
    checkpoints = list_all_checkpoints(ckpt_path + dir)
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(colored(f"Loading model from {latest_checkpoint}", "yellow"))
    config = PeftConfig.from_pretrained(latest_checkpoint)
    model = PeftModel.from_pretrained(llm_model, latest_checkpoint)


def answer(question, rectifier=""):
    prompt = f"### Human: {question}\n### Assistant: {rectifier}"

    batch = tokenizer(prompt,
                      padding=True,
                      truncation=True,
                      return_tensors='pt')
    batch = batch.to(f'cuda:{0}')

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

    ans = generated_text.split("### Assistant:")[-1].replace(rectifier, "").strip()

    return ans


if __name__ == "__main__":
    args = get_args()

    load_inference_model(args.dir)

    if args.mode == "manual":
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
    else:
        test_dataset = get_spm_dataset(phase="finetune", mode="test", with_self_instruct=True)
        for i in range(20):
            text = test_dataset[i]["text"]
            text = text.split("### Human:")[-1].strip()
            texts = text.split("### Assistant:")
            input = texts[0].strip()
            if len(texts) == 1:
                std = ""
            else:
                std = texts[1].strip()
            output = answer(input)

            print("-" * 30)
            print("Input:", input)
            print("Output:", colored(output, "green"))
            print("Standard output:", colored(std, "blue"))
            
            if output[:5] != std[:5]:
                output = answer(input, std[:5])
                print("Rectified output:", std[:5], colored(output, "yellow"))