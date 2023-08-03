from curious_agent import CuriousAgent
from gpt_api import get_llm

from tqdm import tqdm

from data_gen.paths import (uri_attr_to_lang_keyw_prompt_path,
                            prompt_output_path, chatlog_output_path,
                            uri_raw_data_path)

import json
from collections import Counter
import math
import os

num_sample = 20


def extract_attributes_and_values(data):
    items = data["response"]["resultParts"]
    #print(items[0]) # Keywords in Description and ShortDescription
    #print(items[0].keys())
    #for item in items:
    #    if item["Description"] != item["ShortDescription"]:
    #        print(item["Description"], "|||", item["ShortDescription"])
    properties_list = [item["Properties"] for item in items]
    properties = [{k: v
                   for k, v in properties.items()
                   if v}
                  for properties in properties_list]
    # Find the common keys
    keys = [list(p.keys()) for p in properties]
    # 2d list to 1d
    keys = [item for sublist in keys for item in sublist]
    key_count = Counter(keys)

    attributes = list(key_count.keys())
    weights = list(key_count.values())
    values = []
    for attribute in attributes:
        _values = []
        for properties in properties_list:
            if attribute in properties:
                _values.append(properties[attribute])
        values.append(list(set(_values)))

    return attributes, weights, values


def uri_attr_to_lang_keyw(data):
    uri = data["url"]
    uri = uri.replace("https://pdworkbenchapis.azurewebsites.net/search/", "")

    attributes, weights, values = extract_attributes_and_values(data)
    prompts = []
    sum_w = sum(weights)

    num_samples = []
    for i in range(len(attributes)):
        attribute, _values = attributes[i], values[i]
        _values = ", ".join([str(v) for v in _values])
        with open(uri_attr_to_lang_keyw_prompt_path, "r") as handle:
            prompt = handle.read()

        prompt = prompt.replace("{uri}", uri)
        prompt = prompt.replace("{attribute}", attribute)
        prompt = prompt.replace("{values}", _values)
        prompts.append(prompt)

        num_samples.append(int(math.ceil(num_sample * weights[i] / sum_w)))

    return prompts, num_samples


if __name__ == "__main__":
    os.makedirs(prompt_output_path, exist_ok=True)
    os.makedirs(chatlog_output_path, exist_ok=True)
    cnt = 0

    for subdir in ["search1", "search2", "searchString"]:
        file_list = os.listdir(uri_raw_data_path + subdir + "/")
        file_list = [file for file in file_list if file.endswith(".json")]
        file_list.sort()

        for file in tqdm(file_list):
            if not file.endswith(".json"):
                continue

            data = json.load(open(uri_raw_data_path + subdir + "/" + file, "r"))

            prompts, num_samples = uri_attr_to_lang_keyw(data)

            for i in range(len(prompts)):
                prompt_path = prompt_output_path + "uri_" + str(cnt) + ".txt"
                chatlog_path = chatlog_output_path + "uri_" + str(
                    cnt) + "_chatlog.pickle"
                cnt += 1

                with open(prompt_path, 'w') as handle:
                    handle.write(prompts[i])

                agent = CuriousAgent(get_llm(),
                                     prompts[i],
                                     temperature=1,
                                     top_p=0.6)
                for j in range(num_samples[i]):
                    agent.reply()

                agent.dump(chatlog_path)
