from curious_agent import CuriousAgent
from gpt_api import get_llm

from paths import chatlog_output_path

from tqdm import tqdm
import os

num_interaction = 5

file_list = os.listdir(chatlog_output_path)
file_list = [file for file in file_list if file.endswith(".pickle")]
file_list.sort()

os.makedirs(chatlog_output_path, exist_ok=True)

for file in tqdm(file_list):
    print(file)
    store_path = chatlog_output_path + file

    agent = CuriousAgent(get_llm(), store_path)

    completed_interaction = sum(int(m[0] == "user") for m in agent.msgs)

    for i in tqdm(range(completed_interaction, num_interaction)):
        agent.reply()

    agent.dump(store_path)
