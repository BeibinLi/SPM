import json

import requests

from termcolor import colored

url = 'http://gcrsandbox395:5000/question'

while True:
    print("-" * 30)
    question = input("Human: ")
    question = question.strip()

    data = {"question": "### Human: " + question}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    ans = response.json()["answer"]
    print("Bot:", colored(ans, "green"))
