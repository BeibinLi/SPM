import sys, json, pickle, re, pdb

class CuriousAgent:

    def __init__(self, api, system_msg: str, formatter: callable = None, temperature = 0.1, top_p = 0.3):
        self.api = api
        self.system_msg = system_msg
        self.msgs = [("system", system_msg)]
        self.details = []
        self.formatter = formatter
        self.temperature = temperature
        self.top_p = top_p

    def reply(self):
        if len(self.msgs) <= 2:
            prompt = "Go!"
        else:
            prompt = "Thanks! I like your response. Can you try again with a different solution?"
        response = self.api.reply("user",
                             prompt,
                             num_response=1,
                             temperature=self.temperature,
                             top_p=self.top_p,
                             prev_msgs=self.msgs,
                             model="gpt-4")

        self.details.append(response)

        self.msgs.append(("user", prompt))

        rst = self.formatter(response[0])
        rst = str(rst)
        self.msgs.append(("assistant", rst))

    def dump(self, out_loc):
        with open(out_loc, "wb") as f:
            pickle.dump(
                [self.system_msg, self.msgs, self.details,
                 str(self.formatter)], f)

    def load(self, in_loc):
        with open(in_loc, "rb") as f:
            self.system_msg, self.msgs, self.details, self.formatter = pickle.load(
                f)
        self.formatter = eval(self.formatter)


if __name__ == "__main__":
    system_msg = """Generate reading comprehension questions and their answers based on the following content, use the format:

QUESTION: a question goes here
ANSWER: the answer to the question goes here

--- Here are the content ---

Large Language Models (LLMs) have revolutionized Natural Language Processing (NLP) but demand massive GPU resources for training. Lowering the threshold for LLMs training would encourage greater participation from researchers,
benefiting both academia and society. While existing approaches have focused
on parameter-efficient fine-tuning, which tunes or adds a small number of parameters, few have addressed the challenge of tuning the full parameters of LLMs
with limited resources. In this work, we propose a new optimizer, LOw-Memory
Optimization (LOMO), which fuses the gradient computation and the parameter
update in one step to reduce memory usage. By integrating LOMO with existing
memory saving techniques, we reduce memory usage to 10.8% compared to the
standard approach (DeepSpeed solution). Consequently, our approach enables the
full parameter fine-tuning of a 65B model on a single machine with 8×RTX 3090,
each with 24GB memory.1
"""

    formatter = lambda x: re.findall(r"QUESTION: (.*)ANSWER: (.*)", x, re.DOTALL)
    agent = CuriousAgent(system_msg, formatter)

    for i in range(5):
        agent.reply()
        pdb.set_trace()
        agent.dump("test.pickle")
