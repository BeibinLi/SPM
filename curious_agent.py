import json
import pickle
import os
import time
import shutil
import dill    # needed to pickle lambda functions
import tiktoken


class CuriousAgent:

    def __init__(self,
                 api,
                 system_msg: str,
                 formatter: callable = None,
                 temperature=0.1,
                 top_p=0.3,
                 num_response=1,
                 max_token_length=None):
        self.api = api
        self.system_msg = system_msg
        self.msgs = [("system", system_msg)]
        self.details = []
        self.formatter = formatter
        self.temperature = temperature
        self.top_p = top_p
        self.num_response = num_response
        self.max_token_length = max_token_length
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        self.token_length = len(self.encoder.encode(self.msgs[0][1]))

    def reply(self):
        if self.max_token_length and self.token_length > self.max_token_length:
            return

        if len(self.msgs) <= 2:
            prompt = "Go!"
        else:
            prompt = "Thanks! I like your response. Can you try again and come "
            " up with a different response?"
        responses = self.api.reply("user",
                                   prompt,
                                   num_response=self.num_response,
                                   temperature=self.temperature,
                                   top_p=self.top_p,
                                   prev_msgs=self.msgs,
                                   model="gpt-4-32k")

        self.details.append(responses)

        self.msgs.append(("user", prompt))

        if self.formatter is not None:
            responses = [self.formatter(r) for r in responses]

        for rst in responses:
            self.msgs.append(("assistant", rst))
            self.token_length += len(self.encoder.encode(rst))

        return responses

    def dump(self, out_loc):
        with open(out_loc, "wb") as f:
            pickle.dump([
                self.system_msg, self.msgs, [], self.temperature, self.top_p,
                self.num_response,
                dill.dumps(self.formatter)
            ], f)
        with open(out_loc.replace(".pickle", ".json"), "w") as f:
            json.dump([
                self.system_msg, self.msgs, self.details, self.temperature,
                self.top_p, self.num_response
            ],
                      f,
                      indent=4)
        os.makedirs("../cache/", exist_ok=True)
        cache_loc = os.path.join(
            "../cache/", f"{os.path.basename(out_loc)}-{time.time()}.json")
        shutil.copy2(out_loc.replace(".pickle", ".json"), cache_loc)

    def load(self, in_loc):
        with open(in_loc, "rb") as f:
            (self.system_msg, self.msgs, self.details, self.temperature,
            self.top_p, self.num_response, self.formatter) = pickle.load(f)
        self.formatter = dill.loads(self.formatter)
        self.token_length = sum(
            len(m[1]) for m in self.msgs if m[0] == "asisstant")
