import os
import sys
sys.path.append("..")
from SPM.curious_agent import CuriousAgent
from SPM.gpt_api import get_llm
from collections import defaultdict
import tiktoken
from tqdm import tqdm
from gen_prompts import dfs

from paths import *

max_token_length = None
encoder = tiktoken.encoding_for_model("gpt-4")

files = {}

def dfs(path):
    file_list = os.listdir(path)
    for file in file_list:
        new_path = path + file
        if os.path.isdir(new_path):
            dfs(new_path + "/")
        else:
            with open(new_path, mode = "r") as handle:
                content = handle.read()
                if content.replace(" ", "").replace("\n", "") != "": # remove empty files
                    files[new_path[len(raw_data_path):]] = content

def strip_questions(response):
    while True:
        p = response.find("ANSWER")
        if p == -1:
            break

        q = response[p:].find("QUESTION")
        if q == -1:
            q = len(response[p:])
        q += p

        response = response[:p] + response[q:]
    
    return response

def strip_answer_choices(response):
    answer_choices = []
    char_set = set("0123456789:([{ ")
    tmp = response
    while True:
        p = response.find("ANSWER")
        if p == -1:
            break

        q = p + len("ANSWER")
        while q != len(response) and response[q] in char_set:
            q += 1
        
        answer_choices.append(response[q].lower())

        response = response[q:]
    return answer_choices

def verify(response, response_q):
    a1 = strip_answer_choices(response)
    a2 = strip_answer_choices(response_q)

    correct = 0
    for i in range(len(a1)):
        if a1[i] == a2[i]:
            correct += 1
    
    return correct, len(a1)

if __name__ == "__main__":
    with open(prompt_path + "reading_comprehension.md", "r") as handle:
        template = handle.read()
    
    with open(reading_comp_q_path, "r") as handle:
        template_q = handle.read()
    
    cnt, correct, total = 0, 0, 0
    for data_type in ["IFS_code", "IFS_document"]:
        files = {}
        dfs(raw_data_path + data_type + "/")
        file_list = list(files.keys())
        for file in tqdm(file_list):
            print(file)
            with open(raw_data_path + file, mode = "r") as handle:
                content = handle.read()
            
            prompt = template.replace("{content}", content)
            agent = CuriousAgent(api=get_llm(), system_msg=prompt, formatter=None, temperature=1, top_p=0.6, num_response=1, max_token_length=max_token_length)

            response = agent.reply()[0]
            agent.dump(chatlog_output_path + "multichoice" + "_" + str(cnt) + "_chatlog.pickle")

            questions = strip_questions(response)
            
            prompt_q = template_q.replace("{content}", content)
            prompt_q = prompt_q.replace("{questions}", questions)

            agent = CuriousAgent(api=get_llm(), system_msg=prompt_q, formatter=None, temperature=1, top_p=0.6, num_response=1, max_token_length=max_token_length)

            response_q = agent.reply()[0]
            agent.dump(chatlog_output_path + "multichoice" + "_" + str(cnt) + "_verify_chatlog.pickle")
            cnt += 1

            c, t = verify(response, response_q)
            correct += c
            total += t

            print("correct: ", correct, "total: ", total, "accuracy: ", correct / total)
