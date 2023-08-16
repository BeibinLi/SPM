import csv
import pdb
import tempfile
import os
import io
import subprocess
import shlex
import shutil
from termcolor import colored
from utils import (list_files, replace_absolute_with_relative)
from utils import (find_all_substr, trunc_cat, get_file_name)


class AutoExploreSandbox:

    def __init__(
        self,
        dataset_path: str,
    ):
        """
        Wraps the dataset folder.
        Prevents any system commands from ruining the original dataset.

        Args:
        - dataset_path (str): The path to the dataset.
        """
        # use absolute path
        self.working_dir = os.path.abspath(".").replace("\\", "/")
        self.dataset_path = os.path.abspath(dataset_path).replace("\\",
                                                                  "/") + "/"

        # Copy dataset to a temporary directory in the working directory
        self.sandbox_dir = tempfile.mkdtemp(dir=self.working_dir).replace(
            "\\", "/") + "/"

        # ignore hidden files and directories
        def ignore(directory, filenames):
            return [fn for fn in filenames if fn.startswith('.')]

        shutil.copytree(self.dataset_path,
                        self.sandbox_dir,
                        ignore=ignore,
                        dirs_exist_ok=True)
        print(
            colored(f"Data copied to temporary directory: {self.sandbox_dir}",
                    "green"))

        # Setup working dir
        self.original_dir = os.getcwd().replace("\\", "/")
        os.chdir(self.sandbox_dir)

    def __del__(self):
        # Clean up the temporary directory
        # shutil.rmtree(self.sandbox_dir) #shutil.rmtree() may not work
        # properly on Windows
        os.system('rmdir /S /Q "{}"'.format(self.sandbox_dir))
        os.chdir(self.original_dir)

    def run_command(self, cmd: list) -> str:
        """Run a bash command in the dataset sandbox.

        The supported tools are:
        cd, ls, cat, echo, python.

        Args:
            command (list): a

        Returns:
            str: the execution result of the given command. If any errors
                occurred, then just return the error message.
        """
        if cmd[0] in ["cd", "cat"]:
            file = get_file_name(cmd)
            path = os.path.dirname(file) if "." in os.path.basename(
                file) else file
            if path == "":
                path = "."

            original_cwd = os.getcwd()
            try:
                os.chdir(path)
            except Exception as e:
                return "Error: " + str(e)

            cwd = os.getcwd().replace('\\', '/')
            os.chdir(original_cwd)
            if not os.path.abspath(cwd).startswith(
                    os.path.abspath(self.sandbox_dir)):
                return (
                    f"Error: You cannot access files ({cwd}) outside the repo "
                    f"({self.sandbox_dir})! You are now at {os.getcwd()}")

        if cmd[0] not in ["cd", "ls", "cat", "python"]:
            return "Error: You can only run cd, ls, cat, python commands."

        try:
            if cmd[0] == "cd":
                os.chdir(cmd[1])
                return "Success: Now at " + self.relative_cwd()
            else:
                ret = subprocess.run(cmd, encoding="utf-8",
                                     capture_output=True).stdout
                if cmd[0] == "ls":
                    return "Success: The result of ls is:\n" + ret
                elif cmd[0] == "cat":
                    return "Success: The content of " + cmd[
                        1] + " is:\n" + trunc_cat(cmd[1], ret)
                elif cmd[0] == "python":
                    return "Success: The output of python is:\n" + ret
        except Exception as e:
            return "Error: " + str(e)

        # pdb.set_trace()
        raise NotImplementedError    # we should never reach here

    def relative_cwd(self):
        "Return the relative path to the sandbox's root directory."
        # return os.getcwd().replace('\\', '/').replace(self.sandbox_dir, '')
        return os.path.relpath(os.getcwd().replace('\\', '/'),
                               self.sandbox_dir).replace('\\', '/')

    def inject_and_run(self, llm_output: str) -> dict:
        """
        Given the output of LLM, retrieve the target source code and the
        injection snippet. Inject, run, and return the output.

        Args:
        - llm_output (str): The output of LLM, should be in natural language.
            - TARGET_FILE identifies the file to be appended with
            the injection snippet.
            - INJECTION_SNIPPET identifies the code to be injected.
            - COMMAND identifies the command to be run.

        Returns:
        - dict:
            - "stdout": The output of the injected code.
            - "stderr": The error message of the injected code.
            - "changed_files": A dict of changed files,
            key is relative file path,
            value is the content in bytes.
        """
        if "COMMAND" not in llm_output:
            print(colored("Command not found in response.", "yellow"))
            return {
                "stdout":
                    "",
                "stderr":
                    "",
                "changed_files": {},
                "information":
                    f"""You are now at the folder {self.relative_cwd()}"""
            }

        # Step 2: inject the new code into the sandbox
        while llm_output.find("TARGET_FILE:") != -1:
            target_file_start = llm_output.find("TARGET_FILE:") + len(
                "TARGET_FILE:")
            target_file_end = llm_output.find("\n", target_file_start)
            target_file = llm_output[target_file_start:target_file_end].strip()
            llm_output = llm_output[target_file_end:]

            injection_snippet_start = llm_output.find("INJECTION_SNIPPET:")
            injection_snippet_start = llm_output.find(
                "```python\n", injection_snippet_start) + len("```python\n")
            injection_snippet_end = llm_output.find("```",
                                                    injection_snippet_start)
            injection_snippet = llm_output[
                injection_snippet_start:injection_snippet_end]
            llm_output = llm_output[injection_snippet_end:]

            # Read existing content
            target_file_path = os.path.join(self.sandbox_dir, target_file)
            try:
                target_file_content = open(target_file_path, "r").read()
            except Exception as e:
                print(colored(f"Error {e}", "red"))
                target_file_content = ""

            # Write the new content
            target_file_content = target_file_content + "\n" + injection_snippet
            with open(target_file_path, "w") as f:
                f.write(target_file_content)

        print(colored("Injection done.", "green"))

        # extract commands
        bash_start = llm_output.find("COMMAND:")
        bash_start = llm_output.find("```bash\n", bash_start) + len("```bash\n")
        bash_end = llm_output.find("```", bash_start)
        bash = llm_output[bash_start:bash_end].strip()

        # run
        # os.chdir(self.sandbox_dir)
        # ignore all the warnings
        commands = shlex.split(bash)
        commands.insert(1, "-W ignore")
        pdb.set_trace()
        result = subprocess.run(commands, capture_output=True)

        # find all changed files
        original_files = set(list_files(self.dataset_path))
        current_files = set(list_files(self.sandbox_dir))
        changed_files = list(current_files - original_files)

        common_files = current_files.intersection(original_files)

        for file in common_files:
            file = file.replace("\\", "/")

            original_file_path = self.dataset_path + file
            current_file_path = self.sandbox_dir + file

            original_file_content = open(original_file_path, "rb").read()
            current_file_content = open(current_file_path, "rb").read()

            if original_file_content != current_file_content:
                changed_files.append(file)

        print(colored("List of changed files:", "yellow"))
        print(changed_files)

        ret = {
            "stdout":
                result.stdout.decode('utf-8'),
            "stderr":
                replace_absolute_with_relative(result.stderr.decode('utf-8'),
                                               self.sandbox_dir),
            "changed_files": {
                file: open(self.sandbox_dir + file, "rb").read()
                for file in changed_files
            },
            "information":
                f"""You are now at the folder {self.relative_cwd()}"""
        }

        return ret


###


def parse_echo(command: list) -> list:
    """
    Parses an `echo` command string into its constituent parts.

    The function breaks down the echo command into its main components,
    specifically handling redirection using the '>' symbol.

    Args:
    - command (list): A list of strings representing the `echo` command split
        by whitespace.

    Returns:
    - list: A list of parsed components. If the command has a redirection
        (using '>'), the returned list will contain the `echo` command, the
        message to be echoed, the redirection symbol, and the file to which
        the message will be written. If there is no redirection, it will return
        just the `echo` command and the message.

    Example:
    Given the command list:
    ['echo', 'Hello', 'World', '>', 'output.txt']
    The function will return:
    ['echo', '"Hello World"', '>', 'output.txt']

    Note:
    The function assumes that the redirection symbol '>' is always followed by
        the filename
    and that the redirection symbol will only appear once at the end of the
        command.
    """
    assert command[0] == "echo", "The command is not an echo command."
    for i in range(len(command)):
        if command[i].strip().startswith(">"):
            assert i == len(command) - 2
            return [
                "echo", '"' + "".join(command[1:i]) + '"', command[i].strip(),
                command[i + 1]
            ]
    return ["echo", '"' + "".join(command[1:]) + '"']


def extract_command_blocks(response, identifier="```bash"):
    """
    Extracts command blocks encapsulated by markdown code blocks from a given
    response string.

    Parameters:
    - response (str): The input string containing the bash commands enclosed in
        markdown code blocks.
    - identifier (str, optional): The identifier used to recognize the start
        of the bash commands block. Defaults to "```bash", which can also be
        "```python"

    Returns:
    - list: A list of strings, containing the extracted commands.
         Each command is a separate string in the list.

    Example:
    Given the response string:
    '''
    Some text here.
    ```bash
    echo "Hello, World!"
    ls
    ```
    Another text.
    ```bash
    pwd
    ```
    '''
    The function will return:
    ['echo "Hello, World!"\nls', 'pwd']

    Note:
    The function assumes that the end of a bash commands block is marked
    by "```".
    """
    commands = []
    positions = find_all_substr(response, identifier)
    for pos in positions:
        st = pos + len(identifier)
        p = response[st:].find("```") + st
        commands.append(response[st:p].strip())
    return commands


def extract_commands(response: str) -> list:
    """Parse a LLM output to a list of commands, where
    each command is represented in a list of arguments (strs).


    TODO: debug: the csv.reader might not give correct results.
    For instance, `echo hello world > output.txt`


    Args:
        response (str): LLM's response.

    Returns:
        list(list): a 2D list of commands.
    """
    response = response.replace("'", '"')
    bash_commands = extract_command_blocks(response)

    parsed_commands = []

    for bash_command in bash_commands:
        f = io.StringIO(bash_command)
        reader = csv.reader(f,
                            delimiter=' ',
                            quotechar='"',
                            skipinitialspace=True)
        for row in reader:
            if row == []:
                continue
            if row[0] == "echo":
                parsed_commands.append(parse_echo(row))
            else:
                parsed_commands.append(row)

    return parsed_commands


###

if __name__ == "__main__":
    dataset_wrapper = AutoExploreSandbox("../Coffee_Roasting_Dataset")

    llm_output = """TARGET_FILE: visualization/supplier_price.py
INJECTION_SNIPPET:
```python
print("Hello world!")
```
COMMAND:
```bash
python visualization/supplier_price.py --name='Farhunnisa Rajata'
```
"""
    result = dataset_wrapper.inject_and_run(llm_output)
    print(result["stdout"], result["stderr"])
