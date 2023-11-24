import torch

from peft import PeftModel
from transformers import GenerationConfig, AutoTokenizer

from model_utils import calc_probs_log_probs


def compute_policy_gradient(model: PeftModel, tokenizer: AutoTokenizer,
                            generation_config: GenerationConfig,
                            generation_results: list, critic_model: PeftModel,
                            critic_tokenizer: AutoTokenizer) -> list:
    """
    Do one step policy gradient update to the model.

    Args:
    - `model` (PeftModel): the model to be updated
    - `tokenizer` (AutoTokenizer): the tokenizer for `model`
    - `generation_config` (GenerationConfig): the generation config used to
    generate the dialog
    - `generation_results` (list): the generation result, which is a list
    consisting of `batch_size` lists. Each inner list contains dicts in the
    following format:
    {
        "tokens": torch.Tensor,
        "generated_mask": list,
        "cost": float,
        "step": int
    }
    - `critic_model` (PeftModel): the value model to be updated
    - `critic_tokenizer` (AutoTokenizer): the tokenizer for `critic_model`

    Returns:
    - float: the average total cost of the generation results
    """
    use_critic = critic_model is not None

    costs = []

    # Calculate the Q values for all steps
    for generation_result in generation_results:
        # sort the generation results by reversed time order
        generation_result = sorted(generation_result, key=lambda x: -x["step"])
        tot_cost = 0

        for i, step in enumerate(generation_result):
            # update total future cost
            tot_cost += step["cost"]
            step["Q_value"] = tot_cost

        costs.append(tot_cost)

    max_step = 0
    for i, generation_result in enumerate(generation_results):
        # sort the generation results by time order
        generation_result = sorted(generation_result, key=lambda x: x["step"])
        max_step = max(max_step, generation_result[-1]["step"])

    for i in range(max_step):
        # Get the input batch for this step
        Q_values, input_tokens, attention_mask = [], [], []
        generated_mask, prompts = [], []
        for generation_result in generation_results:
            if i < len(generation_result):
                step = generation_result[i]
                Q_values.append(step["Q_value"])
                input_tokens.append(step["tokens"])
                attention_mask.append(step["attention_mask"])
                generated_mask.append(step["generated_mask"])
                prompts.append(step["prompt"])

        Q_values = torch.tensor(Q_values,
                                dtype=torch.float32,
                                device=model.device)

        max_len = max([x.shape[0] for x in input_tokens])

        # Pad to same length
        for i in range(len(input_tokens)):
            input_tokens[i] = torch.cat((torch.full(
                (max_len - input_tokens[i].shape[0],),
                tokenizer.pad_token_id,
                device=input_tokens[i].device), input_tokens[i]))
            attention_mask[i] = torch.cat((torch.full(
                (max_len - attention_mask[i].shape[0],),
                0,
                device=attention_mask[i].device), attention_mask[i]))
            generated_mask[i] = [False] * (
                max_len - len(generated_mask[i])) + generated_mask[i]

        input_tokens = torch.stack(input_tokens)
        attention_mask = torch.stack(attention_mask)
        generated_mask = torch.tensor(generated_mask,
                                      dtype=torch.bool,
                                      device=model.device)

        if use_critic:
            value_inputs = critic_tokenizer.batch_encode_plus(
                prompts,
                truncation=True,
                padding=True,
                max_length=critic_model.config.max_length,
                return_tensors="pt")
            value_inputs = {
                k: v.to(critic_model.device) for k, v in value_inputs.items()
            }
            values = critic_model(**value_inputs).logits.squeeze(-1)
        else:
            values = torch.zeros(len(Q_values),
                                 dtype=torch.float32,
                                 device=model.device)

        # policy gradient uses log probs
        probs_log_probs = calc_probs_log_probs(model,
                                               input_tokens,
                                               attention_mask,
                                               generated_mask,
                                               generation_config,
                                               calc_probs=False,
                                               calc_log_probs=True)
        log_probs = probs_log_probs["log_probs"]

        if len(log_probs) == 0:
            continue

        # Policy network update
        (torch.mean(
            (Q_values - values.detach()) * log_probs) / max_step).backward()

        if use_critic:
            # Value network update
            (torch.nn.MSELoss()(Q_values.to(critic_model.device), values) /
             max_step).backward()

    return sum(costs) / len(costs)
