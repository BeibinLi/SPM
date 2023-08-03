from curious_agent import CuriousAgent
from gpt_api import get_llm

from data_gen.gen_prompts_chatlogs.general import dfs

from data_gen.paths import (raw_data_path, prompt_template_path,
                            reading_comp_q_prompt_path, chatlog_output_path,
                            prompt_output_path)

import tiktoken
from tqdm import tqdm

max_token_length = None
encoder = tiktoken.encoding_for_model("gpt-4")

char_set = set("0123456789:([{ ")
num_set = set("0123456789")

num_interaction = 5

num_questions = 0
files = {}


def strip_questions(response):
    global num_questions
    response = "THINKING" + response
    while True:
        p = response.find("THINKING")    # order: QUESTION, THINKING, ANSWER
        if p == -1:
            break

        q = response[p:].find("QUESTION")
        if q == -1:
            return response[:p]
        q += p

        r = q + len("QUESTION")
        while r != len(response) and response[r] in num_set:
            r += 1

        num_questions += 1
        response = response[:p] + "QUESTION" + str(num_questions) + response[r:]


def strip_answer_choices(response):
    answer_choices = []
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
    with open(prompt_template_path + "reading_comprehension.md", "r") as handle:
        template = handle.read()

    with open(reading_comp_q_prompt_path, "r") as handle:
        template_q = handle.read()

    cnt, correct, total = 0, 0, 0
    for data_type in ["IFS_code", "IFS_document"]:
        files = {}
        dfs(raw_data_path + data_type + "/")
        file_list = list(files.keys())
        pbar = tqdm(file_list)
        for file in pbar:
            with open(raw_data_path + file, mode="r") as handle:
                content = handle.read()

            prompt = template.replace("{content}", content)
            agent = CuriousAgent(api=get_llm(),
                                 system_msg=prompt,
                                 formatter=None,
                                 temperature=1,
                                 top_p=0.6,
                                 num_response=1,
                                 max_token_length=max_token_length)

            prompt_path = prompt_output_path + "multichoice_" + str(
                cnt) + ".txt"
            chatlog_path = chatlog_output_path + "multichoice_" + str(
                cnt) + "_chatlog.pickle"

            with open(prompt_path, 'w') as handle:
                handle.write(prompt)

            response, questions = "", ""
            num_questions = 0
            for i in range(num_interaction):
                t = agent.reply()[0]
                response = response + t
                questions = questions + strip_questions(t)
            agent.dump(chatlog_path)

            prompt_q = template_q.replace("{content}", content)
            prompt_q = prompt_q.replace("{questions}", questions)

            agent = CuriousAgent(api=get_llm(),
                                 system_msg=prompt_q,
                                 formatter=None,
                                 temperature=1,
                                 top_p=0.6,
                                 num_response=1,
                                 max_token_length=max_token_length)

            prompt_q_path = prompt_output_path + "multichoice_" + str(
                cnt) + "_verify.txt"
            chatlog_q_path = chatlog_output_path + "multichoice_" + str(
                cnt) + "_verify_chatlog.pickle"

            with open(prompt_q_path, 'w') as handle:
                handle.write(prompt_q)

            response_q = agent.reply()[0]
            agent.dump(chatlog_q_path)

            c, t = verify(response, response_q)
            correct += c
            total += t

            pbar.set_description("correct: %5d total: %5d accuracy: %5f" %
                                 (correct, total, correct / total))

            cnt += 1
