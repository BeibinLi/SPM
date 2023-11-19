# Small Private Model

## Preparation

### Data

The data contains two parts: repo and task.
Place the repos at `<REPO_PATH>/`, e.g. `data/repos/auto_explore/` and `data/repos/coffee_roasting_dataset`
Place the `.json` task data files at `<TASK_PATH>/*.json`, e.g., `data/tasks/auto_explore.json` and `data/repos/coffee.json`.
There should be a `root` key in the task files specifying which repo it is using.
Then, run
```bash
python -m data_gen.gen_auto_explore_markov --task_file=<TASK_PATH> --repo_dir=<REPO_PATH>
```
to generate data for supervised pretraining.
In the same example,
```bash
python -m data_gen.gen_auto_explore_markov --task_file=data/tasks/ --repo_dir=data/repos/
```
You can generate data for a single task file by
```bash
python -m data_gen.gen_auto_explore_markov --task_file=data/tasks/coffee.json --repo_dir=data/repos/
```


## Run the code

Install packages: `pip install transformers termcolor trl peft bitsandbytes tiktoken fs`


### Supervised pretraining

If you want single-GPU training, simply run
```bash
python supervised_pretrain.py <Args>
```
For example, to run a GPT 2 model, you should
```bash
python supervised_pretrain.py --model_name=gpt2-xl --use_8bit --max_seq_length=1024 --bf16 --max_steps=2000
```

This code uses `accelerate` for parallel training. Make sure to configure `accelerate` before each run if you want multiple-GPU training:

```bash
accelerate config
```
After this, run
```bash
accelerate launch --main_process_port <PORT> supervised_pretrain.py <Args>
```

### RL finetuning for file indentification

After pretraining, run
```bash
python rl_finetune.py --load_dir=<PRETRAIN_DIR> <Other args>
```
This code can only use a single GPU.
For example, to run a GPT2-XL model, you can run
```bash
python rl_finetune.py --load_dir=results/123_supervised_pretrain/checkpoint-2000/ --use_8bit --max_seq_length=1024 --bf16 --use_critic --model_name=gpt2-xl
```
