# Small Private Model

## Preparation


To load a downloaded Llama 2 model, you may need to convert it to Hugging Face. The below instructions could be find at [llama-recipes](https://github.com/facebookresearch/llama-recipes/). Take using 7B as example, make sure you have all the weights in `<WEIGHT_PATH>/7B` and you want to put the converted weights in `<REPO_PATH>/model/llama2/7B`.

```
## Install HuggingFace Transformers from source

pip install "transformers>=4.31"
pip install "tokenizers>=0.13.3"

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


## Run the code

Packages:
`pip install termcolor, trl, peft, bitsandbytes, tiktoken`


If you want single-GPU training, simply run
```bash
python train.py <Args>
```

This code uses `accelerate` for parallel training. Make sure to configure `accelerate` before each run if you want multiple-GPU training:

```bash
accelerate configure
```
After this, run

```bash
accelerate launch --main_process_port <PORT> train.py <Args>
```


`accelerator config` and answer with default choices. When asked "yes/No", always answer: NO, NO, NO

accelerate launch train.py
