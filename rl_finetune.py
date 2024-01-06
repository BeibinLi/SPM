import json
import os
import pdb
import random
import torch

from math import exp
from statistics import mean
from termcolor import colored
from tqdm import tqdm
from transformers import GenerationConfig, HfArgumentParser

from auto_explore_sandbox import RepoCache
from evaluate import batched_answer, calc_Q_values
from experiment_args import ScriptArguments
from functions.cost import StepCost, KeywordCost, NumTokenCost, SynthesizedCost
from model_utils import CriticModel, create_and_prepare_model
from nat_lang_envs.auto_explore import AutoExploreEnv
from trainers import TRAINERS
from utils import (build_curriculum_and_schedule, get_exp_id,
                   load_script_args, load_dataset, ReplayBuffer)

LOG_KEYS = ["Q_value", "prob", "entropy", "cost", "step"]

def calc_avg(arr):
    _arr = arr.copy()
    if isinstance(arr[0], list):
        _arr = sum(_arr, [])
    filtered = [x for x in _arr if x is not None]
    return mean(filtered) if filtered != [] else 0


parser = HfArgumentParser(ScriptArguments)
script_args = load_script_args(parser.parse_args_into_dataclasses()[0])

assert script_args.trainer in TRAINERS, f"Invalid trainer: {script_args.trainer}."

if script_args.disable_dropout:
    if script_args.lora_dropout != 0:
        print(colored(f"disable_dropout is set to True. lora_dropout is overridden to 0.", "yellow"))
        script_args.lora_dropout = 0

# Setup policy network
tokenizer, peft_config, model = create_and_prepare_model(script_args)

exp_id = get_exp_id(script_args.ckpt_path)
output_dir = script_args.ckpt_path + exp_id + "_rl_finetune/"
os.makedirs(output_dir, exist_ok=True)

# Saving the arguments for reference in the future
script_args.dump(os.path.join(output_dir, "setting.yml"))

print(colored("Experiment directory: " + output_dir, "green"))

optimizer = torch.optim.Adam(model.parameters(),
                             lr=script_args.learning_rate,
                             weight_decay=script_args.weight_decay)

if script_args.use_critic:
    # Setup value network, sharing the main body with policy network
    if script_args.shared_critic:
        critic_model = CriticModel(main_model=model,
                                   layer_type=script_args.critic_layer_type)
        critic_optimizer = torch.optim.Adam(critic_model.score.parameters(),
                                            lr=script_args.learning_rate,
                                            weight_decay=script_args.weight_decay)
    else:
        create_and_prepare_model(script_args)
        
else:
    critic_model, critic_optimizer = None, None

# Build the cost function
num_token_cost = NumTokenCost(tokenizer)
keyword_cost = KeywordCost(keywords=["Error", "Warning"], costs=[2, 1])
step_cost = StepCost()
synthesized_cost = SynthesizedCost(
    cost_functions=[num_token_cost, keyword_cost, step_cost], weights=[0, 1, 1])

# Init repo cache
repo_cache = RepoCache(original_root=script_args.repo_dir,
                       dir=script_args.sandbox_dir,
                       file_save_path="changed_files/")

# Build dataset
dataset = load_dataset(script_args.task_file)
dataset, trigger_set = build_curriculum_and_schedule(dataset, script_args)

# Init curriculum
curriculum_idx = -1
cur_dataset = []

# Logs
losses, critic_losses, costs = [], [], []
logs, msgs = [], []
train_logs, critic_train_logs = [], []

replay_buffer = ReplayBuffer(script_args.replay_buffer_size)

# Setup trainer
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

trainer_kwargs = {
    "model": model,
    "tokenizer": tokenizer,
    "optimizer": optimizer,
    "generation_config": generation_config,
    "critic_model": critic_model,
    "critic_optimizer": critic_optimizer,
    "ppo_clip_coef": script_args.ppo_clip_coef,
    "ppo_update_iter": script_args.ppo_update_iter,
    "max_grad_norm": script_args.max_grad_norm,
    "batch_size": script_args.per_device_train_batch_size,
    "entropy_coef": script_args.entropy_coef,
    "gradient_accumulation_steps": script_args.gradient_accumulation_steps,
    "critic_update_freq": script_args.critic_update_freq,
    "critic_update_iter": script_args.critic_update_iter,
}

trainer = TRAINERS[script_args.trainer](**trainer_kwargs)

for iter in (pbar := tqdm(range(script_args.max_steps), desc="Iter")):
    # Move on to the next curriculum
    if iter in trigger_set:
        curriculum_idx += 1
        # Replace dataset
        cur_dataset = dataset[curriculum_idx]
        idx = 0
        random.shuffle(cur_dataset)

    # get current batch
    batch = cur_dataset[idx:idx + script_args.per_device_train_batch_size]
    idx += script_args.per_device_train_batch_size
    if idx >= len(cur_dataset):
        idx = 0
        random.shuffle(cur_dataset)

    cur_logs, cur_msgs = batched_answer(
        env_type=AutoExploreEnv,
        batch=batch,
        model=model,
        tokenizer=tokenizer,
        repo_cache=repo_cache,
        horizon=script_args.horizon,
        generation_config=generation_config,
        cost_function=step_cost,
        leaveout_prob=script_args.leaveout_prob,
        shuffle_action=script_args.shuffle_action,
        easy=script_args.easy,
        first_step=script_args.first_step,
    )

    calc_Q_values(cur_logs, script_args.entropy_coef)

    logs.append(cur_logs)
    msgs.append(cur_msgs)

    cost = mean([log[0]["Q_value"] for log in cur_logs])
    costs.append(cost)

    replay_buffer.add([{
        "data": log,
        "weight": exp(-log[0]["Q_value"] / script_args.horizon)
    } for log in cur_logs if log[0]["Q_value"] <= 0])

    replay_buffer.print()
    
    # Train
    cur_loss, cur_critic_loss = [], []
    datas = [cur_logs, replay_buffer.sample(script_args.per_device_train_batch_size)]
    
    for data in datas:
        train_result = trainer.train(data)
        loss, critic_loss = train_result["loss"], train_result["critic_loss"]
        cur_loss.append(loss)
        cur_critic_loss.append(critic_loss)
    losses.append(cur_loss)
    critic_losses.append(cur_critic_loss)

    # Update tqdm
    avg_cost = calc_avg(costs[-script_args.logging_steps:])
    avg_loss = calc_avg(losses[-script_args.logging_steps:])
    avg_critic_loss = calc_avg(critic_losses[-script_args.logging_steps:])
    pbar.set_description("Cost: %.2f Loss: %.2f Critic Loss: %.2f Iter:" %
                         (avg_cost, avg_loss, avg_critic_loss))

    if (iter + 1) % script_args.save_steps == 0:
        ckpt_path = output_dir + "checkpoint-" + str(iter + 1) + "/"
        os.makedirs(ckpt_path, exist_ok=True)

        # dump the model
        model.save_pretrained(save_directory=ckpt_path)
        tokenizer.save_pretrained(save_directory=ckpt_path)
        if script_args.use_critic:
            critic_path = ckpt_path + "critic/"
            os.makedirs(critic_path, exist_ok=True)
            torch.save(critic_model.score.state_dict(),
                       critic_path + "score.pt")

        # dump the logs
        save_file = []

        for i in range(iter + 1 - script_args.save_steps, iter + 1):
            save_file.append({
                "iter":
                    i,
                "loss":
                    losses[i],
                "critic_loss":
                    critic_losses[i],
                "cost":
                    costs[i],
                "log": [{
                    "batch":
                        b,
                    "detail": [{
                        **{
                            k: lg[k] for k in LOG_KEYS
                        },
                        **{
                            "msg": ms
                        }
                    } for lg, ms in zip(ll, mm)]
                } for b, (ll, mm) in enumerate(zip(logs[i], msgs[i]))],
            })

        with open(ckpt_path + "logs.json", "w") as file:
            json.dump(save_file, file, indent=4)
