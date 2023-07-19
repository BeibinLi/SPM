import sys, os
from gpt_api import get_llm
from utils import *
import re
import csv, io

model = "gpt-4-32k"
#model = "text-davinci-003"

class AutoExploreCopilot():
    def __init__(self, temperature, top_p):
        self.temperature = temperature
        self.top_p = top_p

        self.start_prompt = """You are a helpful AI assistant to help code editing in a large code repo. You need to explore the code repo by sending me system commands: ls, cd, cat, and echo. 

Initially, you are at the root of the repo. Using these commands, your target is to get detailed knowledge of each functionality and class. You need to create a cache file named summarization.txt summarizing the knowledge for future reference. This cache file should be at the root of the repository.

In the future, an independent instance of you may use the summarized cache files to help writing code upon requests of users. You can update the cache files whenever you want by using system commands.

Use the format:
```bash
YOU CODE GOES HERE
```
"""

        self.api = get_llm()
        self.msgs = [("system", self.start_prompt), ("user", "Now lets start!")]
    
    def extract_commands(self, response):
        bash_commands = extract_bash_commands(response)
        print(bash_commands)

        parsed_commands = []
        
        for bash_command in bash_commands:
            f = io.StringIO(bash_command)
            reader = csv.reader(f, delimiter=' ', quotechar='"', skipinitialspace=True)
            for row in reader:
                if row == []:
                    continue
                if row[0] == "echo":
                    parsed_commands.append(parse_echo(row))
                else:
                    parsed_commands.append(" ".join(row))
        
        return parsed_commands

    def act(self):
        response = self.api.reply(
            agent_name=self.msgs[-1][0],
            msg=self.msgs[-1][1],
            num_response=1,
            temperature=self.temperature,
            top_p=self.top_p,
            prev_msgs=self.msgs[:-1],
            model=model
        )[0]
        self.msgs.append(("assistant", response))
        commands = self.extract_commands(response)
        print(response, "\n", commands)
        print("*" * 10)
        for cmd in commands:
            if cmd.startswith("cd"):
                t = os.chdir(cmd[3:].strip())
                print(t)
                self.msgs.append(("user", "Now at: " + os.getcwd()))
            else:
                ret = os.popen(cmd).read()
                if cmd.startswith("ls"):
                    self.msgs.append(("user", "The result of ls is:\n" + ret))
                elif cmd.startswith("cat"):
                    self.msgs.append(("user", "The result of " + cmd + " is:\n" + ret))
                else:
                    self.msgs.append(("user", "Echo success!"))

        if commands == []:
            self.msgs.append(("user", "You didn't give me any command. Please try to further explore the code repo by sending me system commands: ls, cd, cat, and echo. You can append to the cache file. You are now at: " + os.getcwd()))
        print("*" * 20)

if __name__ == "__main__":
    os.chdir("C:/Users/t-rzhou/Desktop/peft-main")
    agent = AutoExploreCopilot(temperature=1, top_p=0.3)
    for i in range(100):
        agent.act()
