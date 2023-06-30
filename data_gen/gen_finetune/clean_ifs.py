import os, sys, re, random, json
import glob
sys.path.append(".")
from data_gen.paths import *

FILE_PATH = finetune_data_path + "ifs_train.jsonl"

if __name__ == "__main__":
    with open(FILE_PATH, "r") as handle:
        content = handle.read()
    
    content = content.replace("Human:", "Human: [IFS]")
    
    with open(FILE_PATH, "w") as handle:
        handle.write(content)
