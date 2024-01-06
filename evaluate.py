import pdb

from peft import PeftModel
from statistics import mean
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, GenerationConfig
from typing import (Optional, Tuple)

from auto_explore_copilot import AutoExploreCopilot
from auto_explore_sandbox import RepoCache
from experiment_args import ScriptArguments
from functions.cost import (AutoExploreCostFunction, StepCost, KeywordCost,
                            NumTokenCost, SynthesizedCost)
from functions.terminate import IdentifyFileTerminate
from model_utils import (create_and_prepare_model,
                         transformer_text_completion)
from utils import load_script_args, load_dataset, build_curriculum_and_schedule



def batched_answer(
    env_type: type,
    batch: list,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    **kwargs,
) -> Tuple[list, list]:
    # init environemtns
    envs, prompts = [], []
    for i in range(len(batch)):
        envs.append(env_type(**kwargs))
        prompts.append(envs[i].reset(batch[i], **kwargs)[0])

    logs, msgs = [[]] * len(batch), [[]] * len(batch)
    step = 0
    while True:
        ret = transformer_text_completion(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            generation_config=envs[0].generation_config)

        prompts, dones = [], []
        for i in range(len(batch)):
            obs, reward, done, _, info = envs[i].step(ret[i]["generation"]["content"])
            prompts.append(obs)
            dones.append(done)
            
            ret[i].update({"cost": -reward, "step": step})
            logs[i].append(ret[i])
            msgs[i].append(info["msg"])
        
        step += 1
        
        if kwargs["first_step"] or all(dones):
            break

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
    costs, successes = [], []

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
        successes += [log[0]["Q_value"] <= 0 for log in logs]

    return mean(costs), mean(successes)

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
    dataset, trigger_set = build_curriculum_and_schedule(dataset, script_args)

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
