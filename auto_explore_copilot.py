import sys, os
from gpt_api import get_llm
from utils import *
import re
import csv, io
import tiktoken
import glob
from config import *
import pickle, json


root = "/home/t-rzhou/raw_data/IFS_code"

class AutoExploreCopilot():
    def __init__(self, temperature, top_p, max_token_length, model, data_path):
        os.chdir(root)

        self.temperature = temperature
        self.top_p = top_p
        self.max_token_length = max_token_length
        self.model = model
        self.data_path = data_path

        
        with open(root + "/long_mem.txt", "a") as f:
            f.write("")

        self.short_mem_path = root + "/short_mem.txt"
        try:
            with open(self.short_mem_path, "r") as f:
                self.short_mem = f.read()
        except:
            self.short_mem = ""

        self.start_prompt = f"""You are a helpful AI assistant to help code editing in a large code repo. You need to explore the code repo by sending me system commands: ls, cd, cat, and echo. 

The tools you can use
1.  Read files by using `cat`. You can read files already in the repo and files that you created. You can only read one file a time to avoid memory and space limits, and you should avoid reading a file multiple times.
2.  Write files by using `echo`. Note that you should not change files that are already in the repo.
3.  List all files with `ls`.
4.  Change directory to a folder with `cd`.

Use the format:
```bash
YOU CODE GOES HERE
```


Note that:
1.  Initially, you are at the root of the repo. Using these commands, your target is to get detailed knowledge of each functionality and class. 
2.  You need to create two cache files named long_mem.txt and short_mem.txt to help you explore.
a.  long_mem.txt must be at the root of the code repo: {os.path.basename(root)}/. It summarizes the knowledge for future reference, e.g., the functionality/purpose of each file/folder. You should update it whenever you finish exploration of a file. Sometimes you will restart, then you may find it helpful to read long_mem.txt to get a sense of what you have done.
b.  {os.path.basename(root)}/short_mem.txt is maintained automatically by a copilot. Make sure to update it whenever necessary. It should be short and concise, indicating what you plan to do, which directory you are at, what directory you have finished exploration so that no future exploration is needed, etc. You only need to include a code block after #UpdateShortMem, containing current short memory and the copilot will override it. It will be given to you every time you restart. Here is an example:

#UpdateShortMem
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
```


3.  You cannot use any other tools or linux commands, besides the ones provided: cd, ls, cat, echo


Here are the tree structure of directories in the repo:
{display_files_recursively(root)}


Here is the information in your short memory. You may need to check it as well as the long memory before you start.
---- short_mem.txt ----
{self.short_mem}

"""
        self.api = get_llm()
        self.msgs = [("system", self.start_prompt), ("user", "Lets start!")]

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
        raise Exception("Invalid echo command: " + str(command))
    
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
            if not cwd.startswith(root):
                self.msgs.append(("user", "Error: You cannot access files outside the repo!"))
                return
            
            if cmd[0] == "echo":
                print(cmd[-1], cwd)
                if cmd[-1].endswith("long_mem.txt") and cwd != root:
                    cmd[-1] = root + "/long_mem.txt"
                    self.msgs.append(("user", "Warning: long_mem.txt must be at the root of repo! The file path is redirected to root."))

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
                    self.msgs.append(("user", "Echo success!"))
        except Exception as e:
            self.msgs.append(("user", "Error: " + str(e)))

    def act(self):
        self.read_count = 0

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
        for cmd in commands:
            self.handle_command(cmd)

        if commands == []:
            self.msgs.append(("user", "Warning: You didn't give me any command. Further explore the code repo by sending me system commands: ls, cd, cat, and echo."))

        if response.find("#UpdateShortMem") != -1:
            mem_blocks = self.extract_bash_commands(response, "```short_mem.txt")
            if mem_blocks == []:
                self.short_mem = ""
                # TODO: assert updated using echo
            else:
                self.short_mem = mem_blocks[0].strip()
                with open(self.short_mem_path, "w") as f:
                    f.write(self.short_mem)
            self.msgs.append(("user", "Short memory updated!"))
        #else:
        #    self.msgs.append(("user", "Warning: You forgot to update short memory."))

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
