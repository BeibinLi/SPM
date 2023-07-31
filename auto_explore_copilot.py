import csv
import glob
import io
import json
import os
import pickle
import re
import sys

import tiktoken

from config import *
from gpt_api import get_llm
from utils import *

root = "/home/t-rzhou/raw_data/IFS_code"
#root = "/home/t-rzhou/RL-for-Combinatorial-Optimization"
#root = "/home/beibinli/MCIO-SCEE-IntelligentFulfillmentService/src/DataProcessing/"

if not os.path.exists(os.path.join(root, "long_mem.txt")):
    open(os.path.join(root, "long_mem.txt"), "w").write("")
if not os.path.exists(os.path.join(root, "short_mem.txt")):
    open(os.path.join(root, "short_mem.txt"), "w").write("")

if not os.path.exists(root):
    print("ROOT not found!")
    exit(1)

class AutoExploreCopilot():
    def __init__(self, temperature, top_p, max_token_length, model, data_path):
        os.chdir(root)

        self.temperature = temperature
        self.top_p = top_p
        self.max_token_length = max_token_length
        self.model = model
        self.data_path = data_path

        self.long_mem_path = root + "/long_mem.txt"
        if not os.path.exists(self.long_mem_path):
            with open(self.long_mem_path, "w"):
                pass

        self.short_mem_path = root + "/short_mem.txt"
        try:
            with open(self.short_mem_path, "r") as f:
                self.short_mem = f.read()
        except:
            self.short_mem = ""

        self.api = get_llm()

        start_prompt = f"""You are a helpful AI assistant to help code editing in a large code repo.
You need to explore the code repo by sending me system commands: ls, cd, cat, and echo. 

Your GOAL is: read the repo, understand what it means and all its files. Then, summarize the knowledge in long_mem.txt.


The tools you can use
1.  Read files by using `cat`. You can read files already in the repo and files that you created. You can only read one file a time to avoid memory and space limits, and you should avoid reading a file multiple times.
2.  Write memory files by using `echo`.
3.  List all files with `ls`.
4.  Change directory to a folder with `cd`.

Use the format:
```bash
YOU CODE GOES HERE
```


Note that:
1.  Initially, you are at the root of the repo. Using these commands, your target is to get detailed knowledge of each functionality and class. 
2.  You need to create two cache files named long_mem.txt and short_mem.txt to help you explore.
    a.  long_mem.txt must be at the root of the code repo: {os.path.basename(root)}/. 
        It summarizes the knowledge for future reference, e.g., the functionality/purpose of each file/folder. 
        You should update it whenever you finish exploration of a file.
          Sometimes you will restart, then you may find it helpful to read long_mem.txt to get a sense of what you have done.
    b.  {os.path.basename(root)}/short_mem.txt is maintained automatically by a copilot. 
        You can write and override it with `cat` command. 
        You should rewrite it whenever you finish exploration of a file.
3. You cannot use any other tools or linux commands, besides the ones provided: cd, ls, cat, echo
4. I am just a bash terminal which can run your commands. I don't have intelligence and can not answer your questions. 
You are all by yourself, and you need to explore the code repo by yourself. 
5. You can use various techniques here, such as summarizing a book, thinking about code logic / architecture design, and performing analyses.
Feel free to use any other abilities, such as planning, executive functioning, etc.
6. Read files one-by-one is not enough. You need to cross reference different files!


----- Sample short_mem.txt file ------
```short_mem.txt
Reasons and Plans:
1. Read file X.py because it is referenced in Y.py
2. I am unclear about the purpose of Z.py, so I will read it.
3. abc.md uses the term "xyz", but I am not sure what it means. So, I will expore "x1.md", "y2.py".
5. The "abc" folder contains a lot of files, so I will explore it later.
6. I have read "abc.py" in the past, but at that time I don't know it is related to "xyz.py". Now, I will re-read abc.py again and check my previous notes in long_mem.txt.
7. Other plans
The files we already read:
  a.py
  b.txt
  xyz/abc.md
Current memory to note:
1. The project is about xyz, with subfolders a, b, c.
2. The xyz folder contains 8 code about the data structures.
3. Folder abc is related to code in folder xyz, which is reference by ...
4. Files in abc folder are read, and they are simple to understand. They represent ...
5. The information about abc is missing, and I am not sure where to find the related information. I will keep reading other files to see if I can find it in the future. But no rush.
6. I found that xyz is about ..., but I haven't written it to long_mem.txt yet. I will need a little bit more time to understand it and then writing to the long_mem.txt
```

----- long_mem.txt information ------
The long_mem.txt file should be a long file with lots of details, and it is append only. 
For instance, you can add details about class, function, helpers, test cases, important variables, and comments into this long-term memory.
You should add as many details as possible here, but not to record duplicate, redundant, trivial, or not useful information. 
You can organize the information in a structured way, e.g., by using a table, or by using a hierarchical structure.


----- tree structure of directories in the repo ------
{display_files_recursively(root)}
"""
        self.msgs = [("system", start_prompt), ("user", "Lets start!")]

        self.encoder = tiktoken.encoding_for_model("gpt-4")
        self.token_length = sum([len(self.encoder.encode(msg[1])) for msg in self.msgs])

        for msg in self.msgs:
            print(colored_string(msg))
    
    def get_cwd(self):
        return os.getcwd().replace('\\', '/').replace(root.replace(os.path.basename(root), ''), '')
    
    def extract_bash_commands(self, response, identifier="```bash"):
        commands = []
        positions = find_all_substr(response, identifier)
        for pos in positions:
            st = pos + len(identifier)
            p = response[st:].find("```") + st
            commands.append(response[st:p].strip())
        return commands

    def parse_echo(self, command):
        for i in range(len(command)):
            if command[i].strip().startswith(">"):
                assert i == len(command) - 2
                return ["echo", '"' + "".join(command[1:i]) + '"', command[i].strip(), command[i+1]]
        return ["echo", '"' + "".join(command[1:]) + '"']
    
    def extract_commands(self, response):
        response = response.replace("'", '"')
        bash_commands = self.extract_bash_commands(response)

        parsed_commands = []
        
        for bash_command in bash_commands:
            f = io.StringIO(bash_command)
            reader = csv.reader(f, delimiter=' ', quotechar='"', skipinitialspace=True)
            for row in reader:
                if row == []:
                    continue
                if row[0] == "echo":
                    parsed_commands.append(self.parse_echo(row))
                else:
                    parsed_commands.append(row)
        
        return parsed_commands

    def handle_command(self, cmd):
        # Test outside repo
        if cmd[0] in ["cd", "cat", "echo"]:
            if cmd[0] == "echo" and len(cmd) != 4:
                self.msgs.append(("user", "Warning: echo command without output file, ignored."))
                return
            
            cmd[-1] = cmd[-1].strip()
            path = os.path.dirname(cmd[-1]) if "." in os.path.basename(cmd[-1]) else cmd[-1]
            if path == "":
                path = "."
            original_cwd = os.getcwd()
            try:
                os.chdir(path)
            except Exception as e:
                self.msgs.append(("user", "Error: " + str(e)))
                return
            cwd = os.getcwd().replace('\\', '/')
            os.chdir(original_cwd)
            if not os.path.abspath(cwd).startswith(os.path.abspath(root)):
                self.msgs.append(("user", "Error: You cannot access files outside the repo!"))
                return
            
            if cmd[0] == "echo":
                # TODO: check if it writes to repo files
                if cmd[-1].endswith("long_mem.txt") and cwd != root:
                    cmd[-1] = root + "/long_mem.txt"
                    self.msgs.append(("user", "Warning: long_mem.txt must be at the root of repo! The file path is redirected to root."))
                if cmd[-1].endswith("short_mem.txt") and cwd != root:
                    cmd[-1] = root + "/short_mem.txt"
                    self.msgs.append(("user", "Warning: short_mem.txt must be at the root of repo, and you do not need to use echo to update it! The file path is redirected to root."))

        try:
            if cmd[0] == "cd":
                os.chdir(cmd[1])
                self.msgs.append(("user", "Now at: " + self.get_cwd()))
            else:
                ret = os.popen(" ".join(cmd)).read()
                if cmd[0] == "ls":
                    self.msgs.append(("user", "The result of ls is:\n" + ret))
                elif cmd[0] == "cat":
                    self.read_count += 1
                    if self.read_count == 1:
                        self.msgs.append(("user", "The content of " + cmd[1] + " is:\n" + ret))
                    else:
                        self.msgs.append(("user", "Warning: You can only read one file at a time. " + cmd[1] + " is ignored."))
                elif cmd[0] == "echo":
                    if cmd[-1].endswith("short_mem.txt"):
                        self.updated_short_mem = True
                    self.msgs.append(("user", "Echo success!"))
        except Exception as e:
            self.msgs.append(("user", "Error: " + str(e)))

    def act(self):
        self.read_count = 0

        msgs_with_short_mem = self.msgs[:-1] + [("assistant", f'---- Current short_mem.txt file. Please update it! ----\n{open(os.path.join(root, "short_mem.txt"), "r").read()}')]

        response = self.api.reply(
            agent_name=self.msgs[-1][0],
            msg=self.msgs[-1][1],
            num_response=1,
            temperature=self.temperature,
            top_p=self.top_p,
            prev_msgs=msgs_with_short_mem,
            model=self.model
        )[0]
        
        self.msgs.append(("assistant", response))

        if self.token_length > self.max_token_length:
            self.dump()
            self.__init__(self.temperature, self.top_p, self.max_token_length, self.model, self.data_path)
            self.msgs.append(("user", "You just restarted the task. You may need to read long memory to pick up the progress."))
            self.token_length += len(self.encoder.encode(self.msgs[-1][1]))
            return
        
        unencoded_pos = len(self.msgs) - 1

        self.updated_short_mem = False

        commands = self.extract_commands(response)
        for cmd in commands:
            self.handle_command(cmd)

        if commands == []:
            self.msgs.append(("user", "Warning: You didn't give me any command. Further explore the code repo by sending me system commands: ls, cd, cat, and echo."))

        if response.find("#UpdateShortMem") != -1:
            mem_blocks = self.extract_bash_commands(response, "```short_mem.txt")
            if mem_blocks != []:
                self.short_mem = mem_blocks[0].strip()
                with open(self.short_mem_path, "w") as f:
                    f.write(self.short_mem)
                self.updated_short_mem = True
        
        if self.updated_short_mem:
            self.msgs.append(("user", "Short memory updated!"))
        else:
            self.msgs.append(("user", "Warning: No update to short memory."))

        # Reset here to incorporate the last assistant messages
        if self.token_length > self.max_token_length:
            self.dump()
            self.__init__(self.temperature, self.top_p, self.max_token_length, self.model, self.data_path)
            self.msgs.append(("user", "You have reached the maximum token length. Now restarted. You may need to read long memory to pick up the progress."))
            self.token_length += len(self.encoder.encode(self.msgs[-1][1]))
            return
        
        self.token_length += sum([len(self.encoder.encode(msg[1])) for msg in self.msgs[unencoded_pos:]])
        
        for msg in self.msgs[unencoded_pos:]:
            print(colored_string(msg))
    
    def dump(self):
        ckpts = os.listdir(self.data_path)
        ckpts = [x.replace(".pickle", "") for x in ckpts]
        ckpt_num_list = [int(x) for x in ckpts if x.isdigit()]
        ckpt_id = max(ckpt_num_list) + 1 if ckpt_num_list != [] else 0
        ckpt = str(ckpt_id).zfill(5)
        out_loc = os.path.join(self.data_path, ckpt + ".pickle")
        with open(out_loc, "wb") as f:
            pickle.dump([
                self.msgs,
                self.temperature,
                self.top_p,
                self.max_token_length,
                self.model,
                self.data_path
            ], f)
        with open(out_loc.replace(".pickle", ".json"), "w") as f:
            json.dump([
                self.msgs,
                self.temperature,
                self.top_p,
                self.max_token_length,
                self.model,
                self.data_path
            ], f, indent=4)

if __name__ == "__main__":
    os.makedirs(auto_explore_data_path, exist_ok=True)
    exp_id = get_exp_id(auto_explore_data_path)
    data_path = os.path.abspath(os.path.join(auto_explore_data_path, exp_id))
    os.makedirs(data_path, exist_ok=True)
    
    agent = AutoExploreCopilot(
        temperature=1,
        top_p=0.3,
        max_token_length=32768 // 2,
        model="gpt-4-32k",
        data_path=data_path
    )
    while True:
        agent.act()
