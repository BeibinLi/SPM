import os
import sys
sys.path.append("..")
from SPM.curious_agent import CuriousAgent
from SPM.gpt_api import get_llm

from tqdm import tqdm

from paths import *

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
