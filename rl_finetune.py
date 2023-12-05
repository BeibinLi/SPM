import json
import os
import random
import torch

from math import exp
from statistics import mean
from termcolor import colored
from tqdm import tqdm
from transformers import (GenerationConfig, HfArgumentParser)

from auto_explore_sandbox import RepoCache
from evaluate import batched_answer
from experiment_args import ScriptArguments
from functions.cost import StepCost, KeywordCost, NumTokenCost, SynthesizedCost
from functions.training import TRAINERS
from model_utils import (CriticModel, load_script_args,
                         create_and_prepare_model)
from utils import build_curriculum, get_exp_id, load_dataset, ReplayBuffer

parser = HfArgumentParser(ScriptArguments)
script_args = load_script_args(parser.parse_args_into_dataclasses()[0])

assert script_args.trainer in TRAINERS, f"Invalid trainer: {script_args.trainer}."

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
    critic_model = CriticModel(model)
    critic_optimizer = torch.optim.Adam(critic_model.score.parameters(),
                                        lr=script_args.learning_rate /
                                        script_args.critic_update_steps,
                                        weight_decay=script_args.weight_decay)
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
                       dir=script_args.sandbox_dir)

# Build dataset
dataset = load_dataset(script_args.task_file)
if script_args.depth_curriculum:
    dataset = build_curriculum(dataset)
else:
    dataset = [dataset]

if script_args.single_batch_data:
    dataset = [dataset[0][:script_args.per_device_train_batch_size]]

step_per_curriculum = script_args.max_steps * 2 // (len(dataset) *
                                                    (len(dataset) + 1))
script_args.max_steps = step_per_curriculum * len(dataset) * (len(dataset) +
                                                              1) // 2

# Iters to add new curriculum
trigger_set = [
    i * (i + 1) // 2 * step_per_curriculum for i in range(len(dataset))
]

# Init curriculum
curriculum_idx = -1
cur_dataset = []

# Logs
losses, costs = [], []
logs, msgs = [], []
train_logs, critic_train_logs = [], []

replay_buffer = ReplayBuffer(script_args.replay_buffer_size)

if script_args.first_curriculum:
    trigger_set = [0]

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
    "critic_update_steps": script_args.critic_update_steps,
}

trainer = TRAINERS[script_args.trainer](**trainer_kwargs)

for iter in (pbar := tqdm(range(script_args.max_steps), desc="Iter")):
    # move on to the next curriculum
    if iter in trigger_set:
        curriculum_idx += 1
        cur_dataset += dataset[curriculum_idx]
        idx = 0
        random.shuffle(cur_dataset)

    # get current batch
    batch = cur_dataset[idx:idx + script_args.per_device_train_batch_size]
    idx += script_args.per_device_train_batch_size
    if idx >= len(cur_dataset):
        idx = 0
        random.shuffle(cur_dataset)

    cur_logs, cur_msgs = batched_answer(
        batch=batch,
        model=model,
        tokenizer=tokenizer,
        repo_cache=repo_cache,
        horizon=script_args.horizon,
        generation_config=generation_config,
        file_save_path="changed_files/",
        cost_function=step_cost,
        leaveout_prob=script_args.leaveout_prob,
        shuffle_action=script_args.shuffle_action,
        easy=script_args.easy,
        first_step=script_args.first_step,
    )

    logs.append(cur_logs)
    msgs.append(cur_msgs)

    cost = mean([log[0]["Q_value"] for log in cur_logs])

    # update the replay buffer
    replay_buffer.add([{
        "data": log,
        "weight": exp(-log[0]["Q_value"] / script_args.horizon)
    } for log in cur_logs])

    loss = trainer.train(cur_logs)
    losses.append(loss)
    costs.append(cost)

    # TODO: add replay buffer training
    # trainer.train(replay_buffer.sample(script_args.per_device_train_batch_size))

    # Update tqdm
    display_costs = [x for x in costs[-script_args.logging_steps:] if x]
    display_losses = [x for x in losses[-script_args.logging_steps:] if x]
    avg_cost = mean(display_costs) if display_costs != [] else 0
    avg_loss = mean(display_losses) if display_losses != [] else 0
    pbar.set_description("Cost: %.2f Loss: %.2f Iter:" % (avg_cost, avg_loss))

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

        for i, log, msg, loss, cost in zip(
                range(iter + 1 - script_args.save_steps,
                      iter + 1), logs, msgs, losses[-script_args.save_steps:],
                costs[-script_args.save_steps:]):
            save_file.append({
                "iter":
                    i,
                "loss":
                    loss,
                "cost":
                    cost,
                "log": [{
                    "batch":
                        b,
                    "detail": [{
                        **{
                            k: lg[k] for k in ["Q_value", "cost", "step"]
                        },
                        **{
                            "msg": m
                        }
                    } for lg, m in zip(ll, mm)]
                } for b, (ll, mm) in enumerate(zip(log, msg))],
            })

        logs, msgs = [], []

        with open(ckpt_path + "logs.json", "w") as file:
            json.dump(save_file, file, indent=4)
