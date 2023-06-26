import pandas as pd
import os
import pdb

ROOT_DIR = "/home/beibinli/cscp_data"
PROMPT_LOCS = [
    "data_gen/prompts/df_question.md", "data_gen/prompts/df_summary.md"
]

# iterate through all the files and subfolders in the directory
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        # only look at the files that end in .json
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(root, file))
        elif file.endswith(".csv"):
            df = pd.read_csv(os.path.join(root, file))
        else:
            continue
        for prompt_loc in PROMPT_LOCS:
            prompt = open(prompt_loc, "r").read()
            prompt = "f'''" + prompt + "'''"
            prompt = eval(prompt)

        # TODO: query the GPT