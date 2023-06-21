import json, random
from collections import Counter
from query_gpt import single_query
from termcolor import colored

uri_attr_to_lang_keyw_prompt_path = "data_gen/prompts/intermediate_prompts/uri_attribute_to_language_keyword.md"
uri_lang_attr_keyw_prompt_path = "data_gen/prompts/intermediate_prompts/uri_language_attribute_to_keyword.md"

def extract_attribute_and_values(data):
    items = data["response"]["resultParts"]
    #print(items[0]) # Keywords in Description and ShortDescription
    #print(items[0].keys())
    #for item in items:
    #    if item["Description"] != item["ShortDescription"]:
    #        print(item["Description"], "|||", item["ShortDescription"])
    properties_list = [item["Properties"] for item in items]
    properties = [{k:v for k, v in properties.items() if v} for properties in properties_list]
    # Find the common keys
    keys = [list(p.keys()) for p in properties]
    # 2d list to 1d
    keys = [item for sublist in keys for item in sublist]
    key_count = Counter(keys)

    attribute = random.choices(list(key_count.keys()), weights=list(key_count.values()))[0]
    values = []
    for properties in properties_list:
        if attribute in properties:
            values.append(properties[attribute])
    
    return attribute, list(set(values))

def uri_attr_to_lang_keyw(data):
    attribute, values = extract_attribute_and_values(data)
    values = ", " .join([str(v) for v in values])
    uri = data["url"]
    uri = uri.replace("https://pdworkbenchapis.azurewebsites.net/search/", "")
    # Ask LLM to generate question
    with open(uri_attr_to_lang_keyw_prompt_path, "r") as handle:
        prompt = handle.read()

    prompt = prompt.replace("{uri}", uri)
    prompt = prompt.replace("{attribute}", attribute)
    prompt = prompt.replace("{values}", values)
    
    return prompt

'''
def uri_lang_attr_to_keyw(data):
    # Extract questions from the answer
    attribute, values = extract_attribute_and_values(data)
    values = ", " .join([str(v) for v in values])
    uri = data["url"]
    uri = uri.replace("https://pdworkbenchapis.azurewebsites.net/search/", "")
    # Ask LLM to generate question
    with open(uri_lang_attr_keyw_prompt_path, "r") as handle:
        prompt = handle.read()

    prompt = prompt.replace("{uri}", uri)
    prompt = prompt.replace("{question}", question)
    prompt = prompt.replace("{attribute}", attribute)
    prompt = prompt.replace("{values}", values)
    
    return prompt
'''

if __name__ == "__main__":
    data = json.load(open("../raw_data/URI/Response15.json", "r"))
    query = uri_attr_to_lang_keyw(data)
    print(query)
    print(single_query(query))
