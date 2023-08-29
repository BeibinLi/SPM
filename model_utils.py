import torch
import glob
import os
from termcolor import colored
import yaml
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)
from peft import PeftConfig, PeftModel

from llama.generation import (Message, Dialog, B_INST, E_INST, B_SYS, E_SYS,
                              SPECIAL_TAGS, UNSAFE_ERROR)


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
    memory_str = [f"### {agent}: {msg}" for agent, msg in messages]
    memory_str = "\n".join(memory_str)

    # TODO: a nicer way to format the prompt
    if question.find("###") < 0:
        prompt = f"### Human: {question}\n"
    else:
        prompt = question
    if question.find("Assistant") < 0:
        prompt += f"### Assistant: {rectifier}"
    else:
        prompt += f"{rectifier}"

    prompt = memory_str + "\n" + prompt

    print(prompt)

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

    return answers


def GPT_msgs_to_Llama_dialog(messages: list) -> Dialog:
    """
    Convert GPT messages to Llama dialog.

    Args:
    - messages (list[dict]): List of messages from GPT, with format:
    [{
        "role": agent_name,
        "content": message_content
    }, ...]
        - `role` taking only ['system', 'user', 'assistant']

    Returns:
    - Dialog: Llama dialog, with same format, but:
        - `role` starts with 'system', then 'user' and 'assistant' alternate
        (u/a/u/a/u...)
    """

    def predict_role(pos, system):
        if system:
            if pos == 0:
                return "system"
            else:
                return ["assistant", "user"][pos % 2]
        else:
            return ["user", "assistant"][pos % 2]

    for message in messages:
        message["role"] = message["role"].lower()
        assert message["role"] in [
            "system", "user", "assistant"
        ], "Role must be in ['system', 'user', 'assistant']."

    pos = 0
    system = messages[0]["role"] == "system"
    content = ""
    dialog = []

    for message in messages:
        role = predict_role(pos, system)
        if message["role"] == role:
            content += "\n" + message["content"]
        else:
            dialog.append(Message(role=role, content=content.strip()))
            pos += 1
            content = message["content"]
    dialog.append(Message(role=role, content=content.strip()))

    return dialog


def Llama_chat_completion(model,
                          tokenizer,
                          dialogs: list,
                          temperature: float = 0.6,
                          top_p: float = 0.9,
                          max_gen_len: int = 4096,
                          logprobs: bool = False) -> list:
    """
    Chat completion for Llama 2.
    """
    prompt_tokens = []
    unsafe_requests = []

    for dialog in dialogs:
        unsafe_requests.append(
            any([
                tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog
            ]))
        if dialog[0]["role"] == "system":
            dialog = [{
                "role":
                    dialog[1]["role"],
                "content":
                    B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
            }] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all([
            msg["role"] == "assistant" for msg in dialog[1::2]
        ]), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        dialog_tokens = sum(
            [
                tokenizer.encode(f"{B_INST} {(prompt['content']).strip()} "
                                 f"{E_INST} {(answer['content']).strip()} ") +
                [tokenizer.eos_token_id] for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        assert (dialog[-1]["role"] == "user"
               ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}")
        prompt_tokens.append(dialog_tokens)

    max_len = max([len(x) for x in prompt_tokens])
    inputs = torch.Tensor([
        x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in prompt_tokens
    ]).long()

    generation_config = GenerationConfig(
        max_length=max_gen_len,
        do_sample=True,
        num_beams=1,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    outputs = model.generate(
        inputs=inputs.to(model.device),
        generation_config=generation_config,
    )[:, inputs.shape[1]:]

    return [{
        "generation": {
            "role": "assistant",
            "content": tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
        }
    } for t, unsafe in zip(outputs, unsafe_requests)]
