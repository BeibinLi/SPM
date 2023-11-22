import glob
import os
import torch
import yaml

from accelerate import Accelerator
from peft import LoraConfig, PeftConfig, PeftModel
from termcolor import colored
from torch import nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)
from transformers.generation.logits_process import LogitsProcessorList

from experiment_args import ScriptArguments
from utils import extract_command_blocks

OVERRIDE_KEYS = ["model_name", "lora_r", "bf16", "fp16", "use_8bit", "use_4bit"]


def load_script_args(script_args: ScriptArguments,
                     override_keys: list = OVERRIDE_KEYS) -> ScriptArguments:
    """
    If `script_args.load_dir` is not None, load the setting.yml from this
    directory if possible. After loading, override the keys in `override_keys`.

    Args:
    - `script_args` (ScriptArguments): the arguments for training.
    - `override_keys` (list): the keys to override.

    Returns:
    - ScriptArguments: The updated script arguments.
    """

    if script_args.load_dir:
        file = os.path.join(script_args.load_dir, "../setting.yml")
        if os.path.exists(file):
            old_script_args = yaml.safe_load(open(file, "r"))

            for key, value in old_script_args.items():
                if key in override_keys and hasattr(script_args, key):
                    setattr(script_args, key, value)
        else:
            print(
                colored(
                    "We cannot find the setting.yml file from the load directory.",
                    "yellow"))

    return script_args


def create_and_prepare_model(
        args: ScriptArguments) -> (AutoTokenizer, PeftConfig, PeftModel):
    """
    Create and prepare model for PEFT training.

    Args:
    - `args` (ScriptArguments): the arguments for training.

    Returns:
    - tuple: A tuple containing:
        - `tokenizer` (AutoTokenizer): Tokenizer associated with the model.
        - `config` (PeftConfig): Configuration of the model.
        - `model` (PeftModel): Loaded model for inference.
    """
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=args.use_8bit,
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

    accelerator = Accelerator()
    local_rank = accelerator.process_index
    device_map = {"": local_rank}

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=args.cache_dir)

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
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

        tokenizer = AutoTokenizer.from_pretrained(args.load_dir)
    else:
        #model = PeftModel(model=base_model, peft_config=peft_config)
        model = base_model
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
            model_max_length=model.config.max_position_embeddings,
            add_prefix_space=False,
        )

    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, peft_config, model


def load_inference_model(
        experiment_dir: str,
        use_original: bool = False) -> (AutoTokenizer, PeftConfig, PeftModel):
    """
    Load the model for inference based on the given experiment directory.

    Args:
    - `experiment_dir` (str): Path to the experiment directory containing the
         model's checkpoint and settings.
    - `use_original` (bool): if True, use the original model rather than
            the fine-tuned model.

    Returns:
    - tuple: A tuple containing:
        - `tokenizer` (AutoTokenizer): Tokenizer associated with the model.
        - `config` (PeftConfig): Configuration of the model.
        - `model` (PeftModel): Loaded model for inference.
    """
    if experiment_dir[-1] != "/":
        experiment_dir += "/"

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
    cache_dir = setting["cache_dir"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=setting["use_4bit"],
        bnb_4bit_quant_type=setting["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=setting["bnb_4bit_compute_dtype"],
        bnb_4bit_use_double_quant=setting["use_nested_quant"],
    )

    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        cache_dir=cache_dir)

    # Load tokenizer from original model
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        cache_dir=cache_dir)
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


def transformer_text_completion(model: PeftModel, tokenizer: AutoTokenizer,
                                prompts: list,
                                generation_config: GenerationConfig) -> list:
    """
    Completion for transformer models.

    Args:
    - `model` (PeftModel): transformer model.
    - `tokenizer` (AutoTokenizer): tokenizer.
    - `prompts` (list): List of prompts.
    - `generation_config` (GenerationConfig): Generation config for the model.

    Returns:
    - list: List of generated messages, with format:
    [{
        "prompt": str,
        "generation": {
            "role": str = "assistant",
            "content": str,
        },
        "tokens": torch.Tensor,
        "generated_mask": list,
    }, ...]
    """
    tokenized = tokenizer(prompts,
                          padding=True,
                          truncation=True,
                          return_tensors="pt")

    inputs = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    max_len = inputs.shape[1]

    outputs = model.generate(inputs=inputs.to(model.device),
                             generation_config=generation_config,
                             attention_mask=attention_mask.to(model.device))

    res = []
    for i, t in enumerate(outputs):
        newly_generated = t[max_len:]
        res.append({
            "prompt":
                prompts[i],
            "generation": {
                "role": "assistant",
                "content": tokenizer.decode(newly_generated)
            },
            "tokens":
                t.cpu(),
            "attention_mask":
                torch.cat((attention_mask[i],
                           torch.tensor([1] * len(newly_generated)))),
            "generated_mask": [False] * max_len + [True] * len(newly_generated),
        })

    return res


def calc_probs_log_probs(
    model: PeftModel,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    generated_mask: torch.Tensor,
    generation_config: GenerationConfig,
    calc_probs: bool = True,
    calc_log_probs: bool = True,
) -> dict:
    """
    Calculate the probability and log probability of the generated tokens with
    gradients.

    Args:
    - `model` (PeftModel): transformer model.
    - `tokens` (torch.Tensor): The complete tokens of original input + output.
    - `attention_mask` (torch.Tensor): The attention mask of `tokens`.
    - `generated_mask` (torch.Tensor): List of generated mask for each position.
    If the position is 1, the token was generated, otherwise it was given by the user.
    - `generation_config` (GenerationConfig): Generation config used to
    generate `tokens`.
    - `calc_probs` (bool): Whether to calculate the probability.
    - `calc_log_probs` (bool): Whether to calculate the log probability.

    Returns:
    - dict: A dictionary of {
        "probs": torch.tensor,
        "log_probs": torch.tensor,
    }
    """
    model_kwargs = {
        "attention_mask": attention_mask,
        "use_cache": True,
    }

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=tokens.shape[1],
        encoder_input_ids=tokens,
        prefix_allowed_tokens_fn=None,
        logits_processor=LogitsProcessorList(),
        model_kwargs=model_kwargs,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
    )
    logits_warper = model._get_logits_warper(generation_config)

    model_inputs = model.prepare_inputs_for_generation(tokens, **model_kwargs)

    outputs = model(**model_inputs, return_dict=True)

    # generate the axis to gather from
    batch_size = outputs.logits.shape[0]
    axis = torch.arange(batch_size, device=model.device)

    probs = torch.ones(batch_size, device=model.device)
    log_probs = torch.zeros(batch_size, device=model.device)

    for pos in range(1, outputs.logits.shape[1]):
        # pre-process distribution
        scores = logits_processor(tokens, outputs.logits[:, pos - 1, :])
        scores = logits_warper(tokens, scores)

        if calc_probs:
            # get probs of current position
            step_probs = nn.functional.softmax(scores, dim=-1)
            step_probs = step_probs[axis, tokens[:, pos]]
        else:
            step_probs = torch.zeros(batch_size, device=model.device)

        if calc_log_probs:
            # get log probs of current position
            step_log_probs = nn.functional.log_softmax(scores, dim=-1)
            step_log_probs = step_log_probs[axis, tokens[:, pos]]
        else:
            step_log_probs = torch.zeros(batch_size, device=model.device)

        probs[generated_mask[:, pos]] *= step_probs[generated_mask[:, pos]]
        log_probs[generated_mask[:, pos]] += step_log_probs[generated_mask[:,
                                                                           pos]]

    return {
        "probs": probs if calc_probs else None,
        "log_probs": log_probs if calc_log_probs else None,
    }


def get_bash_only_generated_masks(logs: list, tokenizer: AutoTokenizer) -> list:
    """
    Get the masks of the generated tokens that are part of a bash.

    Args:
    - `logs` (list): A list of dicts in the
    following format:
    {
        "tokens": torch.Tensor,
        "generated_mask": list,
        ...
    }
    - `tokenizer` (AutoTokenizer): Tokenizer associated with the output.

    Returns:
    - list: A list of generated masks that are part of a bash for each dict.
    """
    ret = []
    for log in logs:
        generated_mask = log["generated_mask"]
        tokens = list(log["tokens"][generated_mask])
        response = tokenizer.decode(tokens)
        blocks = extract_command_blocks(response, only_first=True)[1]

        mask = []
        last_end = 0
        for block in blocks:
            # non bash part
            mask += [False] * len(tokenizer.encode(response[last_end:block[0]]))

            # bash part
            mask += [True] * len(tokenizer.encode(response[block[0]:block[1]]))

            last_end = block[1]

        # remaining non bash part
        mask += [False] * (len(tokens) - len(mask))

        final_mask = []
        j = 0
        for i in range(len(mask)):
            while j < len(generated_mask) and not generated_mask[j]:
                final_mask.append(False)
                j += 1

            final_mask.append(mask[i])
            j += 1

        ret.append(final_mask)

    return ret
