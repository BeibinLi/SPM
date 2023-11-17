import torch

from peft import PeftModel
from transformers import GenerationConfig, AutoTokenizer


def policy_gradient_update(model: PeftModel,
                           generation_config: GenerationConfig,
                           generation_results: list,
                           optimizer: torch.optim.Optimizer,
                           scheduler: torch.optim.lr_scheduler.LambdaLR,
                           critic_model: PeftModel,
                           critic_tokenizer: AutoTokenizer,
                           critic_optimizer: torch.optim.Optimizer,
                           update: bool) -> float:
    """
    Do one step policy gradient update to the model.

    Args:
    - `model` (PeftModel): the model to be updated
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
    - `optimizer` (torch.optim.Optimizer): the optimizer for `model`
    - `scheduler` (torch.optim.lr_scheduler.LambdaLR): the scheduler for `model`
    - `critic_model` (PeftModel): the value model to be updated
    - `critic_tokenizer` (AutoTokenizer): the tokenizer for `critic_model`
    - `critic_optimizer` (torch.optim.Optimizer): the optimizer for `critic_model`
    - `update` (bool): whether to perform gradient update

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

    print(generation_results[0])

    max_step = 0
    for i, generation_result in enumerate(generation_results):
        # sort the generation results by time order
        generation_result = sorted(generation_result, key=lambda x: x["step"])
        max_step = max(max_step, generation_result[-1]["step"])

    for i in range(max_step):
        # Get the input batch for this step
        Q_values, input_tokens, generated_masks, prompts = [], [], [], []
        for generation_result in generation_results:
            if i < len(generation_result):
                step = generation_result[i]
                Q_values.append(step["Q_value"])
                input_tokens.append(step["tokens"])
                generated_masks.append(step["generated_mask"])
                prompts.append(step["prompt"])

        Q_values = torch.tensor(Q_values,
                                dtype=torch.float32,
                                device=model.device)
        input_tokens = torch.stack(input_tokens)
        generated_masks = torch.tensor(generated_masks,
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
        probs_log_probs = model.calc_probs_log_probs(input_tokens,
                                                     generated_masks,
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

    if update:
        optimizer.step()
        scheduler.step()

        if use_critic:
            critic_optimizer.step()

    return sum(costs) / len(costs)


def proximal_policy_optimization_update(
        model: PeftModel, generation_config: GenerationConfig,
        generation_results: list, optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR, critic_model: PeftModel,
        critic_tokenizer: AutoTokenizer,
        critic_optimizer: torch.optim.Optimizer) -> float:
    """
    Do one step proximal policy optimization (PPO) update to the model.

    Args:
    - `model` (PeftModel): the model to be updated
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
    - `optimizer` (torch.optim.Optimizer): the optimizer for `model`
    - `scheduler` (torch.optim.lr_scheduler.LambdaLR): the scheduler for `model`
    - `critic_model` (PeftModel): the value model to be updated
    - `critic_tokenizer` (AutoTokenizer): the tokenizer for `critic_model`
    - `critic_optimizer` (torch.optim.Optimizer): the optimizer for `critic_model`

    Returns:
    - float: the average total cost of the generation results
    """
    costs = []
    optimizer.zero_grad()
    critic_optimizer.zero_grad()

    for generation_result in generation_results:
        # sort the generation results by reversed time order
        generation_result = sorted(generation_result, key=lambda x: -x["step"])
        tot_cost = 0

        value_inputs = critic_tokenizer.batch_encode_plus(
            [x["prompt"] for x in generation_result],
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt")
        value_inputs = {
            k: v.to(critic_model.device) for k, v in value_inputs.items()
        }
        values = critic_model(**value_inputs).logits.squeeze(-1)

        # with torch.no_grad():
        #     probs_log_probs = model.calc_probs_log_probs(input_tokens,
        #                                                  generated_mask,
        #                                                  generation_config,
        #                                                  calc_probs=False,
        #                                                  calc_log_probs=True)
        #     log_probs = probs_log_probs["log_probs"][0]

        Qvalues = []

        # calculate the policy gradient by reversed time order to avoid space
        # explosion
        for i, step in enumerate(generation_result):
            # update total future cost
            tot_cost += step["cost"]
            Qvalues.append(tot_cost)

            # cut off trailing tokens not covered by the mask
            idx = max([0] + [
                x for x in range(len(step["generated_mask"]))
                if step["generated_mask"][x]
            ])
            generation_config.max_length = idx + 1
            input_tokens = torch.tensor(step["tokens"][:idx + 1],
                                        dtype=torch.long,
                                        device=model.device).unsqueeze(0)
            generated_mask = torch.tensor([step["generated_mask"][:idx + 1]],
                                          dtype=torch.bool,
                                          device=model.device)

            # policy gradient uses log probs
            probs_log_probs = model.calc_probs_log_probs(input_tokens,
                                                         generated_mask,
                                                         generation_config,
                                                         calc_probs=False,
                                                         calc_log_probs=True)
            log_probs = probs_log_probs["log_probs"][0]

            if len(log_probs) == 0:
                continue
            # normalize the cost

            # Policy network update
            ((tot_cost - values[i].detach()) * sum(log_probs)).backward()

        costs.append(tot_cost)

        # Value network update
        Qvalues = torch.tensor(Qvalues,
                               dtype=torch.float32,
                               device=critic_model.device)
        torch.nn.MSELoss()(Qvalues, values).backward()

    optimizer.step()
    scheduler.step()

    critic_optimizer.step()

    return sum(costs) / len(costs)
