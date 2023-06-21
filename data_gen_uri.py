import json, random
from collections import Counter


def extract_attribute(data):
    print(data.keys())
    items = data["resultParts"]
    #print(items)
    properties = [item["Properties"] for item in items]
    properties = [{k:v for k, v in p.items() if v} for p in properties]
    # Find the common keys
    keys = [list(p.keys()) for p in properties]
    # 2d list to 1d
    keys = [item for sublist in keys for item in sublist]
    key_count = Counter(keys)

    return random.choice(list(key_count.keys()), weights=list(key_count.values()))


'''
def generate_question(data):
    attribute = extract_attribute(data)
    uri = data["url"]
    uri = uri.replace("https://pdworkbenchapis.azurewebsites.net/search/", "")
    # Ask LLM to generate question
    question_prompt = ""

   

    # Extract questions from the answer

    llm.reply(agent="user", prev_msgs=[("system", "You are a bot to guess what the user is searching from his/her URI/URL inputs.")], question_prompt)

    key_word_prompt = ""

   

    # TODO: extract keywords and remove non-keywords (hard code)




   

    return
'''

if __name__ == "__main__":
    data = json.load(open("data_gen/raw_data/URI/Response15.json", "r"))
    extract_attribute(data["response"])