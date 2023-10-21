# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pdb

import torch
from peft import PeftModel
from transformers import GenerationConfig


def policy_gradient_update(
        model: PeftModel, generation_config: GenerationConfig,
        generation_results: list, optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR) -> float:
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
    - `optimizer` (torch.optim.Optimizer): the optimizer to be used
    - `scheduler` (torch.optim.lr_scheduler.LambdaLR): the scheduler to be used

    Returns:
    - float: the average total cost of the generation results
    """
    costs = []
    optimizer.zero_grad()

    for generation_result in generation_results:
        # sort the generation results by reversed time order
        generation_result = sorted(generation_result, key=lambda x: -x["step"])
        tot_cost = 0
        # calculate the policy gradient by reversed time order to avoid space
        # explosion
        for step in generation_result:
            # update total future cost
            tot_cost += step["cost"]

            # cut off trailing tokens not covered by the mask
            idx = max([0] + [
                x for x in range(len(step["generated_mask"]))
                if step["generated_mask"][x]
            ])
            generation_config.max_length = idx + 2
            input_tokens = torch.tensor(step["tokens"][:idx + 1],
                                        dtype=torch.long,
                                        device=model.device).unsqueeze(0)
            generated_mask = torch.tensor([step["generated_mask"][:idx + 1]],
                                          dtype=torch.bool,
                                          device=model.device)

            # policy gradient uses log probs
            try:
                probs_log_probs = model.calc_probs_log_probs(
                    input_tokens,
                    generated_mask,
                    generation_config,
                    calc_probs=False,
                    calc_log_probs=True)
                print(probs_log_probs)
            except Exception as e:
                print(e)
                pdb.set_trace()
            log_probs = probs_log_probs["log_probs"][0]
            if len(log_probs) == 0:
                continue
            # normalize the cost
            (tot_cost / len(step["tokens"]) * sum(log_probs)).backward()
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(
                            param.grad).any():
                        pdb.set_trace()

        costs.append(tot_cost)

    optimizer.step()
    scheduler.step()

    return sum(costs) / len(costs)
