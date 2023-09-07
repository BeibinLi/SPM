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

import torch
from transformers import (AutoTokenizer, GenerationConfig)
from peft import PeftModel


class AutoExploreCostFunction:
    """
    Cost function for auto explore.
    """
    pass


class NumTokenCost(AutoExploreCostFunction):

    def __init__(self, tokenizer: AutoTokenizer):
        """
        Calculate the number of tokens in the user messages.

        Args:
        - `tokenizer` (AutoTokenizer): The tokenizer used to tokenize the user
        messages.
        """
        self.tokenizer = tokenizer

    def call(self, user_msgs: list) -> int:
        """
        Args:
        - `user_msgs` (list): The user messages.

        Returns:
        - int: The number of tokens in the user messages.
        """
        return sum([len(self.tokenizer.encode(msg[1])) for msg in user_msgs])


class AutoExploreTerminateCriteria:
    """
    Terminate criteria for auto explore.
    """
    pass


class ReachFileTerminate(AutoExploreTerminateCriteria):

    def __init__(self, target_file: str):
        """
        Terminate when the target file is reached.

        Args:
        - `target_file` (str): The target file.
        """

    def call(self, cmds: list) -> bool:
        """
        Args:
        - `cmds` (list): The system commands returned by the assistant.
        """
        # TODO: implement this

        return True


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
        "cost": float
    }
    - `optimizer` (torch.optim.Optimizer): the optimizer to be used
    - `scheduler` (torch.optim.lr_scheduler.LambdaLR): the scheduler to be used

    Returns:
    - float: the average total cost of the generation results
    """
    loss, avg_tot_cost = 0, 0
    for generation_result in generation_results:
        # sort the generation results by reversed time order
        generation_result = sorted(generation_result,
                                   key=lambda x: sum(x["generated_mask"]))
        tot_cost = 0
        # calculate the policy gradient by reversed time order to avoid space
        # explosion
        for step in generation_result:
            # update total future cost
            tot_cost += step["cost"]
            input_tokens = torch.tensor([step["tokens"]],
                                        dtype=torch.long,
                                        device=model.device)
            generated_mask = torch.tensor([step["generated_mask"]],
                                          dtype=torch.bool,
                                          device=model.device)
            # policy gradient uses log probs
            probs_log_probs = model.calc_probs_log_probs(input_tokens,
                                                         generated_mask,
                                                         generation_config,
                                                         calc_probs=False,
                                                         calc_log_probs=True)
            log_probs = probs_log_probs["log_probs"][0]
            loss += tot_cost * log_probs[0]
        avg_tot_cost += tot_cost
    loss /= len(generation_results)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    return avg_tot_cost / len(generation_results)
