import multiprocessing

from multiprocessing import Manager
from peft import PeftModel
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer, GenerationConfig

from auto_explore_copilot import AutoExploreCopilot
from auto_explore_sandbox import RepoCache
from experiment_args import ScriptArguments
from functions.cost import (AutoExploreCostFunction, StepCost, KeywordCost,
                            NumTokenCost, SynthesizedCost)
from functions.terminate import IdentifyFileTerminate
from model_utils import (load_script_args, create_and_prepare_model,
                         transformer_text_completion)
from utils import load_dataset


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
    first_step: bool,
) -> (list, list):
    batch_size = len(batch)

    def roll_out(index, repo_root, sandbox_dir, prompts, rets, logs, msgs,
                 barrier):
        data = batch[index]

        copilot = AutoExploreCopilot(repo_root=repo_root,
                                     sandbox_dir=sandbox_dir,
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
                                     need_output_msgs=False)
        question = f"Find {data['filename']}" if easy else data["question"]
        copilot.set_question(question=question, target_file=data["filename"])

        if first_step:
            copilot.set_answer(data["optimal_path"][1])

        while True:
            if not copilot.is_finished:
                copilot.build_cur_msgs()
                prompts[index] = "\n".join([msg[1] for msg in copilot.cur_msgs])
            else:
                prompts[index] = None

            # Wait for all processes to finish building prompts
            barrier.wait()

            if sum([1 if p else 0 for p in prompts]) == 0:
                # All copilots finished interaction
                break

            barrier.wait()

            if not copilot.is_finished:
                response = copilot.use_lm_ret(rets[index])
                copilot.act_with_response(response)

            if first_step:
                break

        copilot.wrap_up()

        logs[index] = copilot.get_generation_logs()
        msgs[index] = copilot.get_whole_msgs()

        barrier.wait()

    manager = Manager()

    prompts = manager.list([None] * batch_size)
    rets = manager.list([None] * batch_size)
    logs = manager.list([None] * batch_size)
    msgs = manager.list([None] * batch_size)

    barrier = multiprocessing.Barrier(batch_size + 1)

    # Create and start a process for each directory
    processes = []
    for i, data in enumerate(batch):
        if "root" not in data.keys():
            # Only for file_search_coffee.json
            root = "coffee_roasting_dataset"
        else:
            root = data["root"]
        repo_root = repo_cache.cache_repo(root)
        sandbox_dir = repo_cache.get_cache_dir()

        process = multiprocessing.Process(target=roll_out,
                                          args=(i, repo_root, sandbox_dir,
                                                prompts, rets, logs, msgs,
                                                barrier))
        processes.append(process)
        process.start()

    while True:
        barrier.wait()

        if sum([1 if p else 0 for p in prompts]) == 0:
            # All copilots finished interaction
            break

        generation_config = GenerationConfig(
            max_length=max_token_length,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_beams=1,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        lm_return = transformer_text_completion(
            model=model,
            tokenizer=tokenizer,
            prompts=[p for p in prompts if p],
            generation_config=generation_config)

        # Assign the return to the corresponding process
        j = 0
        for i in range(batch_size):
            if prompts[i]:
                rets[i] = lm_return[j]
                j += 1
            else:
                rets[i] = None

        barrier.wait()

        if first_step:
            break

    barrier.wait()

    for process in processes:
        process.join()

    return logs, msgs


def evalutate(
    dataset: list,
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
    first_step: bool,
):
    total_cost = 0

    print(len(dataset))

    for idx in tqdm(
            range(0, len(dataset), script_args.per_device_eval_batch_size)):
        batch = dataset[idx:idx + script_args.per_device_eval_batch_size]

        logs, msgs = batched_answer(
            batch=batch,
            model=model,
            tokenizer=tokenizer,
            repo_cache=repo_cache,
            max_token_length=max_token_length,
            max_new_tokens=max_new_tokens,
            file_save_path=file_save_path,
            cost_function=cost_function,
            leaveout_prob=leaveout_prob,
            shuffle_action=shuffle_action,
            easy=easy,
            first_step=first_step,
        )

        for log in logs:
            total_cost += sum([step["cost"] for step in log])

    return total_cost / len(dataset)


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
    repo_cache = RepoCache(original_root=script_args.repo_dir,
                           dir=script_args.sandbox_dir)

    dataset = load_dataset(script_args.task_file)

    print(
        evalutate(dataset=dataset,
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
                  first_step=script_args.first_step))
