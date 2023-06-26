import os
import sys
sys.path.append("..")
from SPM.curious_agent import CuriousAgent
from SPM.gpt_api import get_llm

from tqdm import tqdm

from paths import *

# def answer_refinement(msgs):
#     qlist = ""
#     for msg in msgs:
#         if msg[0] == "assistant":
#             st = msg[1]
#             q = 0
#             while True:
#                 p = st.find("QUESTION:", q)
#                 if p == -1:
#                     break
#                 q = p
#                 while True:
#                     q += 1
#                     if q == len(st) or st[q] == "\n":
#                         break
#                 qlist += st[p:q] + "\n"
    
#     nmsgs = [
#         msgs[0],
#         ("user", msgs[1][1] + "Now you do not need to generate Q&A, but please answer all the following questions instead. Please copy the problem before its answer to make the reply clear. Here are the questions:\n" + qlist)
#     ]
#     print(nmsgs)
#     print(api.reply("user", resp,
#                   num_response=1,
#                   temperature=0.1,
#                   top_p=0.3,
#                   prev_msgs=nmsgs,
#                   model="gpt-4")[0])

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
