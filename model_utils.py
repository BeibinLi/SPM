import torch
import glob
import os
from termcolor import colored
import yaml
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import PeftConfig, PeftModel


def load_inference_model(
        experiment_dir: str,
        use_original: bool = False) -> (AutoTokenizer, PeftConfig, PeftModel):
    """
    Load the model for inference based on the given experiment directory.

    Args:
        experiment_dir (str): Path to the experiment directory containing the
         model's checkpoint and settings.
        use_original (bool): if True, use the original model rather than
            the fine-tuned model.

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

    model_name = setting["model_name"]    # original base model path

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

    # Load tokenizer from original model
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        model_max_length=8000)
    tokenizer.pad_token = tokenizer.eos_token

    if use_original:
        model = llm_model
        config = None
    else:
        # Load the fine-tuned model latest checkpoint
        # llm_model will be override
        config, model = load_latest_model(llm_model, experiment_dir)

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


def answer(question,
           tokenizer,
           model,
           rectifier="",
           max_new_tokens: int = 300,
           temperature: float = 0.7,
           top_p: float = 0.7,
           num_return_sequences: int = 1,
           messages: list = []):
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

    if len(messages):
        print(colored("TODO!!! Messages are not handled in Llama!", "red"))

    # TODO: a nicer way to format the prompt
    if question.find("###") < 0:
        prompt = f"### Human: {question}\n"
    else:
        prompt = question
    if question.find("Assistant") < 0:
        prompt += f"### Assistant: {rectifier}"
    else:
        prompt += f"{rectifier}"

    batch = tokenizer(prompt,
                      padding=True,
                      truncation=True,
                      return_tensors='pt')
    batch = batch.to(f'cuda:{0}')

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(
            input_ids=batch.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=
            num_return_sequences    # enable for beam search and multiple replies
        )

    answers = []
    for response in output_tokens:
        generated_text = tokenizer.decode(response, skip_special_tokens=True)

        ans = generated_text.split("### Assistant:")[-1].replace(rectifier,
                                                                 "").strip()
        answers.append(ans)

    if num_return_sequences == 1:
        return answers[0]

    return answers
