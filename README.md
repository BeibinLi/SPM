# Small Private Model

## Preparation

### Model

This pipeline can be equipped with both a text completion model and a chat completion model.
Specifically, we choose GPT 2 for text completion and Llama 2 for chat completion.

To use a GPT 2 model, set `--model_name=gpt2-xl` when running.

To use a Llama 2 model, set `--model_name=model/llama2/xB` after downloading and converting a Llama 2 model.
To load a downloaded Llama 2 model, you may need to convert it to Hugging Face.
The below instructions could be find at [llama-recipes](https://github.com/facebookresearch/llama-recipes/).
Take using 7B as example, make sure you have all the weights in `<WEIGHT_PATH>/7B` and you want to put the converted weights in `<REPO_PATH>/model/llama2/7B`.

```
## Install HuggingFace Transformers from source

```bash
pip install "transformers>=4.34"
pip install "tokenizers>=0.13.3"
```

git clone git@github.com:huggingface/transformers.git
# Or: git clone http://github.com/huggingface/transformers
cd transformers
pip install protobuf
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir <WEIGHT_PATH> --model_size 7B --output_dir <REPO_PATH>/model/llama2/7B


# Note: you may need to change the "main" function in "src/transformers/models/llama/convert_llama_weights_to_hf.py"
```

If you received tokenizer warnings, you can try to change the line to `tokenizer = tokenizer_class(input_tokenizer_path, legacy=False)`

Then set `model_name = "model/llama2/7B"` in `config.py`.

**Note:** To use xB-chat, you still need to put the weights under the folder `<WEIGHT_PATH>/xB` and call with `--model_size xB`.


### Data

Place the `json` data file at `data/file_search_coffee.json`.
Then, run
```bash
python -m data_gen.gen_auto_explore_markov
```
to generate data for supervised pretraining.


## Run the code

Install packages: `pip install termcolor trl peft bitsandbytes tiktoken`


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
