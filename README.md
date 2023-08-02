# Small Private Model

## Preparation

To load a downloaded Llama 2 model, you may need to convert it to Hugging Face. The below instructions could be find at [llama-recipes](https://github.com/facebookresearch/llama-recipes/). Take using 7B as example, make sure you have all the weights in `<WEIGHT_PATH>/7B` and you want to put the converted weights in `<REPO_PATH>/model/llama2/7B`.

```
## Install HuggingFace Transformers from source
pip freeze | grep transformers ## verify it is version 4.31.0 or higher

git clone git@github.com:huggingface/transformers.git
cd transformers
pip install protobuf
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir <WEIGHT_PATH> --model_size 7B --output_dir <REPO_PATH>/model/llama2/7B
```

Then set `model_name = "model/llama2/7B"` in `config.py`.

**Note:** To use xB-chat, you still need to put the weights under the folder `<WEIGHT_PATH>/xB` and call with `--model_size xB`.


## Run the code

If you want single-GPU training, simply run
```
python train.py
```

This code uses `accelerate` for parallel training. Make sure to configure `accelerate` before each run if you want multiple-GPU training:
```
accelerate configure
```
After this, run
```
accelerate launch train.py <Args>
```
