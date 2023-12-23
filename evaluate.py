import pdb

from peft import PeftModel
from statistics import mean
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, GenerationConfig

from auto_explore_copilot import AutoExploreCopilot
from auto_explore_sandbox import RepoCache
from experiment_args import ScriptArguments
from functions.cost import (AutoExploreCostFunction, StepCost, KeywordCost,
                            NumTokenCost, SynthesizedCost)
from functions.terminate import IdentifyFileTerminate
from model_utils import (load_script_args, create_and_prepare_model,
                         transformer_text_completion)
from utils import load_dataset, build_curriculum


def batched_answer(
    batch: list,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    repo_cache: RepoCache,
    horizon: int,
    generation_config: GenerationConfig,
    file_save_path: str,
    cost_function: AutoExploreCostFunction,
    leaveout_prob: float,
    shuffle_action: bool,
    easy: bool,
    first_step: bool,
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
                               horizon=horizon,
                               generation_config=generation_config,
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

        if first_step:
            copilots[-1].set_answer(data["optimal_path"][1])

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

        if first_step:
            break

    for copilot in copilots:
        copilot.wrap_up()

    # get logs
    logs, msgs = [], []
    for copilot in copilots:
        logs.append(copilot.get_generation_logs())
        msgs.append(copilot.get_whole_msgs())

    return logs, msgs


def evalutate(
    dataset: list,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    repo_cache: RepoCache,
    horizon: int,
    batch_size: int,
    generation_config: GenerationConfig,
    file_save_path: str,
    cost_function: AutoExploreCostFunction,
    leaveout_prob: float,
    shuffle_action: bool,
    easy: bool,
    first_step: bool,
):
    costs = []

    for idx in tqdm(
            range(0, len(dataset), batch_size)):
        batch = dataset[idx:idx + batch_size]

        logs, msgs = batched_answer(
            batch=batch,
            model=model,
            horizon=horizon,
            tokenizer=tokenizer,
            repo_cache=repo_cache,
            generation_config=generation_config,
            file_save_path=file_save_path,
            cost_function=cost_function,
            leaveout_prob=leaveout_prob,
            shuffle_action=shuffle_action,
            easy=easy,
            first_step=first_step,
        )

        calc_Q_values(logs)

        costs += [log[0]["Q_value"] for log in logs]

    return mean(costs)

def calc_Q_values(logs, entropy_coef=0):
    for log in logs:
        tot_cost = 0
        for i in range(len(log) - 1, -1, -1):
            tot_cost += log[i]["cost"] - entropy_coef * log[i]["entropy"]
            # tot_cost += log[i]["cost"] + entropy_coef * log[i]["log_prob"]
            log[i]["Q_value"] = tot_cost

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = load_script_args(parser.parse_args_into_dataclasses()[0])

    # Setup policy network
    tokenizer, peft_config, model = create_and_prepare_model(script_args)
    model.eval()

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
    repo_cache = RepoCache(original_root=script_args.repo_dir,
                           dir=script_args.sandbox_dir)

    dataset = load_dataset(script_args.task_file)
    if script_args.depth_curriculum:
        dataset = build_curriculum(dataset, merge_first_two=False, first_k=3)
    else:
        dataset = [dataset]
    if script_args.first_curriculum:
        dataset = dataset[:1]

    generation_config = GenerationConfig(
        max_length=script_args.max_seq_length,
        max_new_tokens=script_args.max_new_tokens,
        do_sample=True,
        num_beams=1,
        temperature=script_args.temperature,
        top_p=script_args.top_p,
        top_k=script_args.top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    for i in range(len(dataset)):
        depth = dataset[i][0]["filename"].count("/")
        print(
            "Depth %d :" % depth,
            evalutate(dataset=dataset[i],
                      model=model,
                      tokenizer=tokenizer,
                      repo_cache=repo_cache,
                      horizon=script_args.horizon,
                      batch_size=script_args.per_device_eval_batch_size,
                      generation_config=generation_config,
                      file_save_path="changed_files/",
                      cost_function=step_cost,
                      leaveout_prob=script_args.leaveout_prob,
                      shuffle_action=script_args.shuffle_action,
                      easy=script_args.easy,
                      first_step=script_args.first_step))
