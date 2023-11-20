import json
import os
import random
import torch

from tqdm import tqdm
from transformers import (GenerationConfig, HfArgumentParser, GPT2Tokenizer,
                          GPT2ForSequenceClassification)

from auto_explore_sandbox import RepoCache
from evaluate import batched_answer
from experiment_args import ScriptArguments
from functions.cost import StepCost, KeywordCost, NumTokenCost, SynthesizedCost
from functions.training import compute_policy_gradient
from model_utils import (load_script_args, create_and_prepare_model)
from utils import build_curriculum, get_exp_id

parser = HfArgumentParser(ScriptArguments)
script_args = load_script_args(parser.parse_args_into_dataclasses()[0])

exp_id = get_exp_id(script_args.ckpt_path)
output_dir = script_args.ckpt_path + exp_id + "_rl_finetune/"
os.makedirs(output_dir, exist_ok=True)

# Saving the arguments for reference in the future
script_args.dump(os.path.join(output_dir, "setting.yml"))

# Setup policy network
tokenizer, peft_config, model = create_and_prepare_model(script_args)

optimizer = torch.optim.Adam(model.parameters(), lr=script_args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

if script_args.use_critic:
    # Setup value network
    critic_model = GPT2ForSequenceClassification.from_pretrained(
        'gpt2', num_labels=1).cuda()
    critic_model.config.pad_token_id = critic_model.config.eos_token_id
    critic_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    critic_tokenizer.pad_token = critic_tokenizer.eos_token
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=1e-4)
else:
    critic_model, critic_tokenizer, critic_optimizer = None, None, None

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
if script_args.depth_curriculum:
    dataset = build_curriculum(dataset)
else:
    dataset = [dataset]

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
total_cost = 0
costs = []
msgs = []

for iter in (pbar := tqdm(range(script_args.max_steps), desc="Iter")):
    if iter % script_args.gradient_accumulation_steps == 0:
        optimizer.zero_grad()
        if script_args.use_critic:
            critic_optimizer.zero_grad()

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

    generation_config = GenerationConfig(
        max_length=script_args.max_seq_length,
        max_new_tokens=script_args.max_new_tokens,
        do_sample=True,
        num_beams=1,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    logs, _msgs = batched_answer(
        batch=batch,
        model=model,
        tokenizer=tokenizer,
        repo_cache=repo_cache,
        max_token_length=script_args.max_seq_length,
        max_new_tokens=script_args.max_new_tokens,
        file_save_path="changed_files/",
        cost_function=step_cost,
        leaveout_prob=script_args.leaveout_prob,
        shuffle_action=script_args.shuffle_action,
        easy=script_args.easy,
    )

    msgs += _msgs

    # update the model
    cost = compute_policy_gradient(
        model=model,
        generation_config=generation_config,
        generation_results=logs,
        optimizer=optimizer,
        scheduler=scheduler,
        critic_model=critic_model,
        critic_tokenizer=critic_tokenizer,
        critic_optimizer=critic_optimizer,
    )
    costs.append(cost)
    total_cost += cost

    pbar.set_description("Cost: %.2f Iter:" % (total_cost / (iter + 1)))

    if (iter + 1) % script_args.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       script_args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        if script_args.use_critic:
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(),
                                           script_args.max_grad_norm)
            critic_optimizer.step()

    if (iter + 1) % script_args.save_steps == 0:
        ckpt_path = output_dir + "checkpoint-" + str(iter + 1) + "/"
        os.makedirs(ckpt_path, exist_ok=True)

        # dump the model
        model.save_pretrained(save_directory=ckpt_path)
        tokenizer.save_pretrained(save_directory=ckpt_path)
        if script_args.use_critic:
            critic_path = ckpt_path + "critic/"
            os.makedirs(critic_path, exist_ok=True)
            critic_model.save_pretrained(save_directory=critic_path)
            critic_tokenizer.save_pretrained(save_directory=critic_path)

        # dump the messages
        with open(ckpt_path + "msgs.json", "w") as f:
            f.write("\n".join(
                [json.dumps(line) for msg in msgs for line in msg]))
        msgs = []

        # dump the costs
        with open(ckpt_path + "costs.json", "w") as f:
            f.write("\n".join([str(cost) for cost in costs]))
        costs = []
