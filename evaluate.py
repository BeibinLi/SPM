import json
import os

from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, HfArgumentParser, AutoTokenizer

from auto_explore_copilot import AutoExploreCopilot
from auto_explore_sandbox import RepoCache
from experiment_args import ScriptArguments
from functions.cost import (AutoExploreCostFunction, StepCost, KeywordCost,
                            NumTokenCost, SynthesizedCost)
from functions.terminate import IdentifyFileTerminate
from model_utils import (load_script_args, create_and_prepare_model,
                         transformer_text_completion)


def batched_answer(
    batch: list,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    repo_cache: RepoCache,
    max_token_length: int,
    max_new_tokens: int,
    file_save_path: str,
    cost_function: AutoExploreCostFunction,
    leaveout_prob: float,
    shuffle_action: bool,
    easy: bool,
) -> (list, list):
    # init copilots
    copilots = []
    for i in range(len(batch)):
        data = batch[i]
        if "root" not in data.keys():
            # Only for file_search_coffee.json
            root = "coffee_roasting_dataset"
        else:
            root = data["root"]

        copilots.append(
            AutoExploreCopilot(repo_root=repo_cache.cache_repo(root),
                               sandbox_dir=repo_cache.cache_dir,
                               temperature=1,
                               top_p=1,
                               max_token_length=max_token_length,
                               max_new_tokens=max_new_tokens,
                               file_save_path=file_save_path,
                               interaction_type="train",
                               model_type="local",
                               model=model,
                               tokenizer=tokenizer,
                               cost_function=cost_function,
                               terminate_criteria=IdentifyFileTerminate(
                                   data["filename"]),
                               leaveout_prob=leaveout_prob,
                               shuffle_action=shuffle_action,
                               easy=easy,
                               need_output_msgs=False))
        question = f"Find {data['filename']}" if easy else data["question"]
        copilots[-1].set_question(question=question,
                                  target_file=data["filename"])

    while True:
        prompts = []
        for i in range(len(batch)):
            if not copilots[i].is_finished:
                copilots[i].build_cur_msgs()
                prompts.append("\n".join(
                    [msg[1] for msg in copilots[i].cur_msgs]))

        if len(prompts) == 0:
            break

        ret = transformer_text_completion(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            generation_config=copilots[0].generation_config)

        j = 0
        for i in range(len(batch)):
            if not copilots[i].is_finished:
                response = copilots[i].use_lm_ret(ret[j])
                copilots[i].act_with_response(response)
                j += 1

    for copilot in copilots:
        copilot.wrap_up()

    # get logs
    logs, msgs = [], []
    for copilot in copilots:
        logs.append(copilot.get_generation_logs())
        msgs.append(copilot.get_whole_msgs())

    return logs, msgs


def evalutate():
    for idx in tqdm(
            range(0, len(dataset), script_args.per_device_eval_batch_size)):
        dataset[idx:idx + script_args.per_device_eval_batch_size]

        GenerationConfig(
            max_length=script_args.max_seq_length,
            max_new_tokens=script_args.max_new_tokens,
            do_sample=True,
            num_beams=1,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = load_script_args(parser.parse_args_into_dataclasses()[0])

    # Setup policy network
    tokenizer, peft_config, model = create_and_prepare_model(script_args)

    # Sampling unchanged logits
    temperature = 1
    top_p = 1

    # Build the cost function
    num_token_cost = NumTokenCost(tokenizer)
    keyword_cost = KeywordCost(keywords=["Error", "Warning"], costs=[100, 20])
    step_cost = StepCost()
    synthesized_cost = SynthesizedCost(
        cost_functions=[num_token_cost, keyword_cost], weights=[1, 1])

    # Init repo cache
    repo_cache = RepoCache(script_args.sandbox_dir)

    # Build dataset
    if script_args.task_file.endswith(".json"):
        task_files = [script_args.task_file]
    else:
        task_files = [
            os.path.join(script_args.task_file, f)
            for f in os.listdir(script_args.task_file)
            if f.endswith(".json")
        ]
    dataset = []
    for task_file in task_files:
        dataset += json.load(open(task_file, "r"))

    # Logs
    total_loss = 0
    losses = []
    msgs = []
