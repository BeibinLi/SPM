import glob
import os
import pdb
import yaml
import torch
from peft import PeftConfig, PeftModel
from termcolor import colored
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from utils import get_spm_dataset
import argparse


def get_args() -> argparse.Namespace:
    """Parses command line arguments for the inference script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        type=str,
                        default=None,
                        required=True,
                        help="Dir to load model")
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "manual"],
        help="Mode: 'auto' for auto testing on random samples, 'manual' "
        "for manual input")
    return parser.parse_args()


def load_inference_model(
        experiment_dir: str) -> (AutoTokenizer, PeftConfig, PeftModel):
    """
    Load the model for inference based on the given experiment directory.

    Args:
        experiment_dir (str): Path to the experiment directory containing the
         model's checkpoint and settings.

    Returns:
        tuple: A tuple containing:
            - tokenizer (AutoTokenizer): Tokenizer associated with the model.
            - config (PeftConfig): Configuration of the model.
            - model (PeftModel): Loaded model for inference.
    """
    setting_file = os.path.join(experiment_dir, "setting.yml")
    if os.path.exists(setting_file):
        setting = yaml.safe_load(open(setting_file, "r"))
    else:
        print(
            colored(
                "We cannot find the setting.yml file. So loading the "
                "default_setting.yml from the root of the repo.", "yellow"))
        setting = yaml.safe_load(open("default_setting.yml", "r"))

    model_name = setting["model_name"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=setting["use_4bit"],
        bnb_4bit_quant_type=setting["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=setting["bnb_4bit_compute_dtype"],
        bnb_4bit_use_double_quant=setting["use_nested_quant"],
    )

    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        quantization_config=bnb_config,
        trust_remote_code=True)

    config, model = load_latest_model(llm_model, experiment_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, config, model


def load_latest_model(llm_model, experiment_dir):
    """
    Loads the latest model checkpoint from the given experiment directory.

    Args:
        llm_model (AutoModelForCausalLM): Base model for loading checkpoint.
        experiment_dir (str): Path to the experiment directory containing the
             model's checkpoint.

    Returns:
        tuple: A tuple containing:
            - config (PeftConfig): Configuration of the loaded model.
            - model (PeftModel): Loaded model from the latest checkpoint.
    """
    latest_checkpoint = max(glob.glob(
        os.path.join(experiment_dir, "checkpoint-*")),
                            key=os.path.getctime)
    print(colored(f"Loading model from {latest_checkpoint}", "yellow"))
    config = PeftConfig.from_pretrained(latest_checkpoint)
    model = PeftModel.from_pretrained(llm_model, latest_checkpoint)
    return config, model


def answer(question, tokenizer, model, rectifier=""):
    """
    Generates an answer for the given question using the provided model and
    tokenizer.

    Args:
        question (str): User's question.
        tokenizer (AutoTokenizer): Tokenizer associated with the model.
        model (PeftModel): Model to generate the answer.
        rectifier (str, optional): Text used for rectifying the model's output.
            Defaults to an empty string.

    Returns:
        str: Generated answer from the model.
    """
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

    ans = generated_text.split("### Assistant:")[-1].replace(rectifier,
                                                             "").strip()
    return ans


if __name__ == "__main__":
    args = get_args()

    tokenizer, config, model = load_inference_model(args.dir)

    if args.mode == "manual":
        while True:
            print("-" * 30)
            question = input("Human: ").strip()

            if not question:
                continue
            if question == "quit":
                break
            elif question == "pdb":
                pdb.set_trace()
            elif question == "load":
                # Reload the model and config
                config, model = load_latest_model(model, args.dir)
            else:
                ans = answer(question, tokenizer, model)
                print("Bot:", colored(ans, "green"))
    else:
        test_dataset = get_spm_dataset(phase="finetune",
                                       mode="test",
                                       with_self_instruct=True)
        for i in range(20):
            text = test_dataset[i]["text"].split("### Human:")[-1].strip()
            input, std = text.split(
                "### Assistant:") if "### Assistant:" in text else (text, "")
            input, std = input.strip(), std.strip()
            output = answer(input, tokenizer, model)

            print("-" * 30)
            print("Input:", input)
            print("Output:", colored(output, "green"))
            print("Standard output:", colored(std, "blue"))

            if output[:5] != std[:5]:
                output = answer(input, tokenizer, model, std[:5])
                print("Rectified output:", std[:5], colored(output, "yellow"))
