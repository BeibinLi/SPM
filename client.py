import json

import requests

url = 'http://gcrsandbox321:5000/question'
data = {'question': 'Can you tell me the ModelNumber of the C-2030 server'}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())
