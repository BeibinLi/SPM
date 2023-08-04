from data_gen.paths import (prompt_output_path, uri_raw_data_path,
                            chatlog_output_path, uri_df_prompt_path)

import pandas as pd
import os
import io
import tiktoken
from tqdm import tqdm

max_token_length = 3000
encoder = tiktoken.encoding_for_model("gpt-4")


def save_content(path, content, prompt):
    #if len(encoder.encode(content)) > max_token_length:
    #    return

    with open(path, 'w') as handle:
        handle.write(content)


num_samples = 500
size_sample = 10

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

prompts = {}

df = pd.read_parquet(uri_raw_data_path + "Where-Use/WhereUsed.parquet")

os.makedirs(chatlog_output_path, exist_ok=True)

file_list = os.listdir(uri_df_prompt_path)
for file in file_list:
    if not file.endswith(".md"):
        continue
    with open(uri_df_prompt_path + file, mode="r") as handle:
        template = handle.read()

    for i in tqdm(range(num_samples)):
        prompt = template
        ndf = df.sample(size_sample)

        output = io.StringIO()
        print(ndf, file=output)
        prompt = prompt.replace("{df.sample()}", output.getvalue())

        prompt = prompt.replace("{df.keys()}", " ".join(ndf.keys().tolist()))

        output = io.StringIO()
        print(ndf.describe(include='all'), file=output)
        prompt = prompt.replace("{df.describe(include='all')}",
                                output.getvalue())

        output = io.StringIO()
        print(ndf.shape, file=output)
        prompt = prompt.replace("{df.shape}", output.getvalue())

        save_content(
            prompt_output_path + file[:-len(".md")] + "_" + str(i) + ".txt",
            prompt, file)
