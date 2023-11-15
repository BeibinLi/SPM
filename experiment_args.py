from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on model location, data location,
    what their capacity, features, etc.
    """

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=1024)
    max_new_tokens: Optional[int] = field(default=1)
    model_name: Optional[str] = field(
    #default="model/llama2/7B-chat",
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging "
                    "Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    load_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Where to load the pretrained models. None for no loading. "
                "latest for latest checkpoint. directory for loading from a "
                "directory."
        })
    task_file: Optional[str] = field(
        default="data/tasks/file_search_coffee.json",
    #default="data/tasks/",
        metadata={
            "help": "The path to the task file. Could be a directory or a "
                    "specific file. All files should contain the path of "
                    "associated repositories."
        },
    )
    repo_dir: Optional[str] = field(
        default="data/repos/",
        metadata={
            "help": "The path to the directory containing the repositories."
        },
    )
    ckpt_path: Optional[str] = field(
        default="results/",
        metadata={
            "help": "The location to save the experiment checkpoints. It "
                    " should be the folder with all experiments."
        })
    use_8bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate 8bit precision base model loading"},
    )
    use_4bit: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of training epochs for the reward model."
        },
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than "
                    "cosine, and has advantage for analysis"
        },
    )
    max_steps: int = field(
        default=5000,
        metadata={"help": "How many optimizer "
                          "update steps to take"})
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Fraction of "
                          "steps to do a warmup for"})
    group_by_length: bool = field(
        default=False,
        metadata={
            "help": "Group sequences into batches with same length. Saves "
                    "memory and speeds up training considerably."
        },
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint "
                          "every X updates steps."})
    save_total_limit: int = field(
        default=50,
        metadata={
            "help": "Limit the total amount of checkpoints. "
                    "Deletes the older checkpoints."
        })
    logging_steps: int = field(default=10,
                               metadata={"help": "Log every X updates steps."})
    cache_dir: Optional[str] = field(
        default="model/",
        metadata={"help": "Where to store the pretrained models."})
    use_critic: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether use critic in RL finetuning."})
    easy: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether use easy task (file finding)."})
    leaveout_prob: Optional[float] = field(
        default=0.5,
        metadata={
            "help":
                "The probability to leave out unrelated files when training."
        })
    depth_curriculum: Optional[bool] = field(
        default=False,
        metadata={
            "help":
                "Whether use depth curriculum: sort the target files by their"
                " depth, and train in increasing order."
        })

    def load(self, yaml_file: str):
        with open(yaml_file, 'r') as file:
            yaml_data = yaml.safe_load(file)

        for key, value in yaml_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def dump(self, filename: str):
        with open(filename, 'w') as file:
            yaml.dump(self.__dict__, file)
