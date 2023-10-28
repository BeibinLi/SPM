import torch

from peft import PeftModel
from transformers import GenerationConfig, AutoTokenizer


def policy_gradient_update(model: PeftModel,
                           generation_config: GenerationConfig,
                           generation_results: list,
                           optimizer: torch.optim.Optimizer,
                           scheduler: torch.optim.lr_scheduler.LambdaLR,
                           value_model: PeftModel,
                           value_tokenizer: AutoTokenizer,
                           value_optimizer: torch.optim.Optimizer) -> float:
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
    - `value_model` (PeftModel): the value model to be updated
    - `value_tokenizer` (AutoTokenizer): the tokenizer for `value_model`
    - `value_optimizer` (torch.optim.Optimizer): the optimizer for `value_model`

    Returns:
    - float: the average total cost of the generation results
    """
    use_critic = value_model is not None

    costs = []
    optimizer.zero_grad()

    if use_critic:
        value_optimizer.zero_grad()

    for generation_result in generation_results:
        # sort the generation results by reversed time order
        generation_result = sorted(generation_result, key=lambda x: -x["step"])
        tot_cost = 0

        if use_critic:
            value_inputs = value_tokenizer.batch_encode_plus(
                [x["prompt"] for x in generation_result],
                truncation=True,
                padding=True,
                max_length=1024,
                return_tensors="pt")
            value_inputs = {
                k: v.to(value_model.device) for k, v in value_inputs.items()
            }
            values = value_model(**value_inputs).logits.squeeze(-1)
        else:
            values = torch.zeros(len(generation_result),
                                 dtype=torch.float32,
                                 device=model.device)

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

        if use_critic:
            # Value network update
            Qvalues = torch.tensor(Qvalues,
                                   dtype=torch.float32,
                                   device=value_model.device)
            torch.nn.MSELoss()(Qvalues, values).backward()

    optimizer.step()
    scheduler.step()

    if use_critic:
        value_optimizer.step()

    return sum(costs) / len(costs)
