import sys, os
from gpt_api import get_llm
from utils import *
import re
import csv, io
import tiktoken
import glob

root = "C:/Users/t-rzhou/Desktop/peft-main"

class AutoExploreCopilot():
    def __init__(self, temperature, top_p, max_token_length, model):
        self.temperature = temperature
        self.top_p = top_p
        self.max_token_length = max_token_length
        self.model = model

        self.short_mem_path = root + "/short_mem.txt"
        try:
            with open(self.short_mem_path, "r") as f:
                self.short_mem = f.read()
        except:
            self.short_mem = ""

        self.start_prompt = f"""You are a helpful AI assistant to help code editing in a large code repo. You need to explore the code repo by sending me system commands: ls, cd, cat, and echo. 

The tools you can use
1.  Read files by using `cat`. You can read files already in the repo and files that you created. Please read one file a time to avoid memory and space limits.
2.  Write files by using `echo` or `cat`. Note that you should not change files that are already in the repo.
3.  List all files with `ls`.
4.  Change directory to a folder with `cd`.

Use the format:
```bash
YOU CODE GOES HERE
```


Note that:
1.  Initially, you are at the root of the repo. Using these commands, your target is to get detailed knowledge of each functionality and class. 
2.  You need to create two cache files named long_mem.txt and short_mem.txt to help you explore. These cache files must be at the root of the repository {os.path.basename(root)}.
a.  long_mem.txt summarizes the knowledge for future reference. You can read it whenever and however you like. This file is used when an independent instance of you is asked to help write code upon requests of users.
b.  short_mem.txt is maintained automatically by a copilot. It should be short and concise, indicating what you plan to do, which directory you are at, etc. You only need to include a code block of current short memory at the end of your response and the copilot will override it. It will be given to you every time you restart.

Sample short_mem.txt
```short_mem.txt
Planning: 
1.	TODO 1
2.	TODO 2
Reasons:
1.	Reason 1
2.	Reason 2
Current memory to note:
1.	Memory 1
2.	Memory 2
3.	…
4.	At most 10 memories here!
```


3.  You cannot use any other tools or linux commands, besides the ones provided: cd, ls, cat, echo


Here are the tree structure of directories in the repo:
{display_files_recursively(root)}

Now you are at {self.get_cwd()}


Here is the information in your short memory. You may need to check it as well as the long memory before you start.
---- short_mem.txt ----
{self.short_mem}

"""

        self.api = get_llm()
        self.msgs = [("system", self.start_prompt), ("user", "Lets start!")]

        self.encoder = tiktoken.encoding_for_model("gpt-4")
        self.token_length = sum([len(self.encoder.encode(msg[1])) for msg in self.msgs])
    
    def get_cwd(self):
        return os.getcwd().replace('\\', '/').replace(root.replace(os.path.basename(root), ''), '')
    
    def extract_commands(self, response):
        bash_commands = extract_bash_commands(response)

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
            model=self.model
        )[0]
        self.msgs.append(("assistant", response))
        unencoded_pos = len(self.msgs) - 1

        commands = self.extract_commands(response)
        print(response)
        print("*" * 10)
        for cmd in commands:
            if cmd.startswith("cd"):
                temp_path = os.path.join(os.getcwd(), cmd[3:].strip())
                os.chdir(cmd[3:].strip())
                self.msgs.append(("user", "Now at: " + self.get_cwd()))
            else:
                ret = os.popen(cmd).read()
                if cmd.startswith("ls"):
                    self.msgs.append(("user", "The result of ls is:\n" + ret))
                elif cmd.startswith("cat"):
                    self.msgs.append(("user", "The result of " + cmd + " is:\n" + ret))
                else:
                    self.msgs.append(("user", "Echo success!"))

        if commands == []:
            self.msgs.append(("user", "You didn't give me any command. Please try to further explore the code repo by sending me system commands: ls, cd, cat, and echo."))

        if response.find("#UpdateShortMem") != -1:
            self.short_mem = extract_bash_commands(response, "```short_mem.txt")[0].strip()
            with open(self.short_mem_path, "w") as f:
                f.write(self.short_mem)
            self.msgs.append(("user", "Short memory updated!"))
        else:
            self.msgs.append(("user", "You forgot to update short memory."))
        
        self.token_length += sum([len(self.encoder.encode(msg[1])) for msg in self.msgs[unencoded_pos:]])
        if self.token_length > self.max_token_length:
            self.__init__(self.temperature, self.top_p, self.max_token_length, self.model)
            self.msgs.append(("user", "You have reached the maximum token length. Send fewer commands in a single response. Now restarted."))
            self.token_length += len(self.encoder.encode(self.msgs[-1][1]))

        for msg in self.msgs[unencoded_pos+1:]:
            print(msg[1])
        print("*" * 20)

if __name__ == "__main__":
    os.chdir(root)
    agent = AutoExploreCopilot(temperature=1, top_p=0.3, max_token_length=32768 // 2, model="gpt-4-32k")
    for i in range(100):
        agent.act()
