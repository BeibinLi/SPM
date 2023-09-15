import torch
from torch import nn
import glob
import os
import copy
import inspect
from termcolor import colored
import yaml
import pdb
from utils import extract_command_blocks
from accelerate import Accelerator
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig)
from transformers.generation.logits_process import (LogitsProcessorList)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList, validate_stopping_criteria)
from peft import PeftConfig, LoraConfig, PeftModel
from llama.generation import (Message, Dialog, B_INST, E_INST, B_SYS, E_SYS,
                              SPECIAL_TAGS, UNSAFE_ERROR)

from experiment_args import ScriptArguments


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
    else:
        #model = PeftModel(model=base_model, peft_config=peft_config)
        model = base_model

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              trust_remote_code=True,
                                              cache_dir=args.cache_dir)
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


def answer(question: str,
           tokenizer: AutoTokenizer,
           model: PeftModel,
           rectifier: str = "",
           max_new_tokens: int = 300,
           temperature: float = 0.7,
           top_p: float = 0.7,
           num_return_sequences: int = 1,
           messages: list = []):
    """
    Generates an answer for the given question using the provided model and
    tokenizer.

    Args:
    - `question` (str): User's question.
    - `tokenizer` (AutoTokenizer): Tokenizer associated with the model.
    - `model` (PeftModel): Model to generate the answer.
    - `rectifier` (str, optional): Text used for rectifying the model's output.
            Defaults to an empty string.

    Returns:
    - str: Generated answer from the model.
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
    - `messages` (list[dict]): List of messages from GPT, with format:
    [(agent_name, message_content), ...] or
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

    if not isinstance(messages[0], dict):
        # convert the first format to the second
        messages = [{
            "role": agent_name,
            "content": message_content
        } for agent_name, message_content in messages]

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
    dialog.append(
        Message(role=predict_role(pos, system), content=content.strip()))

    return dialog


def build_Llama_prompt_from_dialogs(
        tokenizer: AutoTokenizer,
        dialogs: list,
        check_last_user: bool = True) -> (list, list):
    """
    Build Llama prompt from dialogs.

    Args:
    - `tokenizer` (AutoTokenizer): Llama tokenizer.
    - `dialogs` (list[Dialog]): List of dialogs, with format:
    [{
        "role": agent_name,
        "content": message_content
    }, ...]
        - `role` taking only ['system', 'user', 'assistant']
        - `role` starts with 'system', then 'user' and 'assistant' alternate
        (u/a/u/a/u...)
    - `check_last_user` (bool): Whether to check last message from user.

    Returns:
    - tuple: A tuple containing:
        - `prompt_tokens` (list): List of prompt tokens.
        - `unsafe_requests` (list): List of bools indicating whether the
        request is unsafe.
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
        if check_last_user:
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
        if dialog[-1]["role"] == "user":
            dialog_tokens += tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}")
        prompt_tokens.append(dialog_tokens)

    return prompt_tokens, unsafe_requests


def Llama_chat_completion(model: PeftModel, tokenizer: AutoTokenizer,
                          dialogs: list,
                          generation_config: GenerationConfig) -> list:
    """
    Chat completion for Llama 2.

    Args:
    - `model` (PeftModel): Llama model.
    - `tokenizer` (AutoTokenizer): Llama tokenizer.
    - `dialogs` (list[Dialog]): List of dialogs, with format:
    [{
        "role": agent_name,
        "content": message_content
    }, ...]
        - `role` taking only ['system', 'user', 'assistant']
        - `role` starts with 'system', then 'user' and 'assistant' alternate
        (u/a/u/a/u...)
    - `generation_config` (GenerationConfig): Generation config for the model.

    Returns:
    - list: List of generated messages, with format:
    [{
        "generation": {
            "role": str,
            "content": str
        },
        "tokens": torch.Tensor,
        "generated_mask": list
    }, ...]
    """
    assert len(
        dialogs) == 1, "Currently do not support batched dialogs for training."

    prompt_tokens, unsafe_requests = build_Llama_prompt_from_dialogs(
        tokenizer=tokenizer, dialogs=dialogs)

    # left-padding
    max_len = max([len(x) for x in prompt_tokens])
    inputs = torch.Tensor([([tokenizer.pad_token_id] * (max_len - len(x)) + x)
                           for x in prompt_tokens]).long()

    outputs = model.generate(
        inputs=inputs.to(model.device),
        generation_config=generation_config,
    )

    def remove_trailing_eos(tokens):
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] != tokenizer.eos_token_id:
                return tokens[:i + 1]

    # outputs contain an EOS token in the end
    # remove it when decoding
    res = []
    for t, unsafe in zip(outputs, unsafe_requests):
        newly_generated = t[max_len:]
        res.append({
            "generation": {
                "role":
                    "assistant",
                "content":
                    tokenizer.decode(remove_trailing_eos(newly_generated))
                    if not unsafe else UNSAFE_ERROR,
            },
            "tokens": t,
            "generated_mask": [False] * max_len + [True] * len(newly_generated),
        })

    return res


# A new function for PeftModel to support calc prob and log prob WITH GRADIENTS
def calc_probs_log_probs(
    self,
    inputs: torch.Tensor,
    generated_mask: torch.Tensor,
    generation_config: GenerationConfig,
    calc_probs: bool = True,
    calc_log_probs: bool = True,
    **kwargs,
) -> dict:
    """
    A member function of the Llama model.
    Calculate the probability and log probability of the generated tokens.

    Args:
    - `inputs` (torch.Tensor): The generated tokens.
    - `generated_mask` (torch.Tensor): List of generated mask for each position.
    If the position is 1, the token was generated, otherwise it was given by the user.
    - `generation_config` (GenerationConfig): Generation config used to
    generate `inputs`.
    - `calc_probs` (bool): Whether to calculate the probability.
    - `calc_log_probs` (bool): Whether to calculate the log probability.

    Returns:
    - dict: A dictionary of {
        "probs": list of probabilities for each generated text span,
        "log_probs": list of log probabilities for each generated text span,
    }
    """
    # 1. Handle `generation_config` and kwargs that might update it,
    # and validate the `.generate()` call
    self._validate_model_class()

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(
        **kwargs)    # All unused kwargs must be model kwargs
    generation_config.validate()
    self._validate_model_kwargs(model_kwargs.copy())

    if (generation_config.pad_token_id is None
            and generation_config.eos_token_id is not None):
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs)

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs[
        "output_hidden_states"] = generation_config.output_hidden_states
    # decoder-only models with inputs_embeds forwarding must use caching
    # (otherwise we can't detect whether we are generating the first new token
    # or not, and we only want to use the embeddings for the first new token)

    if model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(
        inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get(
            "attention_mask", None
    ) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs[
            "attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id,
                generation_config.eos_token_id)

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop(
        "input_ids")

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get(
        "max_length") is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = (generation_config.max_new_tokens +
                                        input_ids_length)
    self._validate_generated_length(generation_config, input_ids_length,
                                    has_default_max_length)

    # 8. prepare distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=None,
        logits_processor=LogitsProcessorList(),
        model_kwargs=model_kwargs,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
    )

    # 9. prepare stopping criteria
    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=StoppingCriteriaList())

    # 11. prepare logits warper
    logits_warper = self._get_logits_warper(generation_config)

    # 12. expand input_ids with `num_return_sequences` additional sequences per batch
    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_return_sequences,
        is_encoder_decoder=self.config.is_encoder_decoder,
        **model_kwargs,
    )

    # init values
    max_length = generation_config.max_length
    if max_length is not None:
        stopping_criteria = validate_stopping_criteria(stopping_criteria,
                                                       max_length)
    eos_token_id = self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    torch.tensor(eos_token_id).to(
        input_ids.device) if eos_token_id is not None else None

    # compared to the original sample():
    # output_scores = True
    # output_attentions = False
    # output_hidden_states = False
    # return_dict_in_generate = True

    # prepare model inputs
    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # forward pass to get next token
    # TODO: everytime, this line will take some memory. Try to resolve.
    outputs = self(**model_inputs, return_dict=True)

    # generate the axis to gather from
    batch_size = outputs.logits.shape[0]
    axis = torch.arange(batch_size, device=self.device)

    probs = [[] for _ in range(batch_size)]
    log_probs = [[] for _ in range(batch_size)]

    accumulated_probs = torch.ones(batch_size, device=self.device)
    accumulated_log_probs = torch.zeros(batch_size, device=self.device)
    accumulating = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

    for pos in range(1, outputs.logits.shape[1]):
        # pre-process distribution
        scores = logits_processor(input_ids, outputs.logits[:, pos - 1, :])
        scores = logits_warper(input_ids, scores)

        if calc_probs:
            # get probs of current position
            step_probs = nn.functional.softmax(scores, dim=-1)
            step_probs = step_probs[axis, input_ids[:, pos]]
        else:
            step_probs = torch.zeros(batch_size, device=self.device)

        if calc_log_probs:
            # get log probs of current position
            step_log_probs = nn.functional.log_softmax(scores, dim=-1)
            step_log_probs = step_log_probs[axis, input_ids[:, pos]]
        else:
            step_log_probs = torch.zeros(batch_size, device=self.device)

        # accumulate probs
        for i in range(batch_size):
            if not generated_mask[i][pos]:
                if accumulating[i]:
                    if calc_probs:
                        probs[i].append(accumulated_probs[i].clone())
                        pdb.set_trace()
                    if calc_log_probs:
                        log_probs[i].append(accumulated_log_probs[i].clone())
                    accumulated_probs[i] = 1
                    accumulated_log_probs[i] = 0
                    accumulating[i] = False

        accumulating |= generated_mask[:, pos]
        step_probs[~accumulating] = 1
        step_log_probs[~accumulating] = 0
        accumulated_probs *= step_probs
        accumulated_log_probs += step_log_probs

    # update final probs and log probs
    for i in range(batch_size):
        if accumulating[i]:
            if calc_probs:
                probs[i].append(accumulated_probs[i])
            if calc_log_probs:
                log_probs[i].append(accumulated_log_probs[i])

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
        tokens = log["tokens"][generated_mask]
        response = tokenizer.decode(tokens)
        blocks = extract_command_blocks(response)[1]

        mask = []
        last_end = 0
        for block in blocks:
            # non bash part
            mask += [False] * len(tokenizer.encode(response[last_end:block[0]]))

            # bash part
            mask += [True] * len(tokenizer.encode(response[block[0]:block[1]]))

            last_end = block[1]

        # remaining non bash part
        mask += [False] * len(tokenizer.encode(response[last_end:]))

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
