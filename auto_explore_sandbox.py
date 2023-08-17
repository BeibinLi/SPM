import tempfile
import os
import subprocess
import shutil
import pdb
import random
import string
from hashlib import sha256
from termcolor import colored
from utils import (list_files, replace_absolute_with_relative, get_target_dir,
                   trunc_cat, get_file_name)


class AutoExploreSandbox:

    def __init__(
        self,
        dataset_path: str,
        password: str = "",
        private_files: list = [],
    ):
        """
        Wraps the dataset folder.
        Prevents any system commands from ruining the original dataset.

        Args:
        - dataset_path (str): The path to the dataset.
        """
        # Use absolute path
        self.working_dir = os.path.abspath(".").replace("\\", "/") + "/"
        self.dataset_path = os.path.abspath(dataset_path).replace("\\",
                                                                  "/") + "/"

        # Copy dataset to a temporary directory in the working directory
        self.sandbox_dir = tempfile.mkdtemp(dir=self.working_dir).replace(
            "\\", "/") + "/"

        # Store the hashed password for identity verification
        self._sandbox_id = ''.join(
            random.choices(string.ascii_uppercase + string.digits, k=10))
        self._hashed_password = self._hash_password(password)
        self.private_files = private_files

        # Ignore hidden files and directories
        def ignore(directory, filenames):
            return [fn for fn in filenames if fn.startswith('.')]

        shutil.copytree(self.dataset_path,
                        self.sandbox_dir,
                        ignore=ignore,
                        dirs_exist_ok=True)
        print(
            colored(f"Data copied to temporary directory: {self.sandbox_dir}",
                    "green"))

        # Checkpoint cwd to avoid outside changes
        self.cwd = self.sandbox_dir

    def __del__(self):
        # Clean up the temporary directory
        # If it is windows, run 'rmdir'
        # otherwise, run rm -rf
        if os.name == "nt":
            os.system('rmdir /S /Q "{}"'.format(self.sandbox_dir))
        else:
            os.system('rm -rf "{}"'.format(self.sandbox_dir))

    def _hash_password(self, password: str) -> str:
        return sha256(password.encode('utf-8') + self._sandbox_id).hexdigest()

    def safety_check(self, cmd: [list, str], password: str) -> str:
        """
        Return "SAFE" iff the cmd is safe to run.


        Otherwise, return "DANGER" and why it is not safe to run.
        """
        # First check if password is correct
        if self._hash_password(password) != self._hashed_password:
            return "DANGER: Wrong password!"

        # Then check if the target file is private
        return "SAFE"

    def run_command(self, cmd: [list, str], password: str) -> str:
        """Wrapper function for self.run_command().
        Run a bash command in the dataset sandbox.

        The supported tools are:
        cd, ls, cat, echo, python.

        Args:
        - cmd (list or str): a single command splitted into arguments or
             a string of the command.

        Returns:
        - str: the execution result of the given command. If any errors
        occurred, then just return the error message.
        """
        is_safe = self.safety_check(cmd, password)
        if is_safe != "SAFE":
            return "Sorry, your command is not safe to run! Because:\n" + is_safe

        # Restore to the checkpointed cwd
        _cwd = os.getcwd()
        os.chdir(self.cwd)

        if type(cmd) is list:
            ret = self._run_command(cmd)
        elif type(cmd) is str:
            ret = self._run_raw_command(cmd)

        # Checkpoint cwd
        self.cwd = os.getcwd().replace("\\", "/") + "/"
        os.chdir(_cwd)

        return ret

    def _run_raw_command(self, cmd: str) -> str:
        """
        Args:
        - cmd (str): a command block, which could contain several lines of
            commands.

        Returns:
        - str: the execution result of the given command. If any errors
        occurred, then just return the error message.
        """

        # Run the command
        rst_msg = ""
        try:
            wrapped_cmd = cmd + "\n\npwd\n"
            result = subprocess.run(wrapped_cmd,
                                    shell=True,
                                    capture_output=True)
            stdout = result.stdout.decode('utf-8')

            error = result.stderr.decode('utf-8')
            if len(error):
                rst_msg += "Error encountered:\n" + error + "\n\n"
                os.chdir(self.sandbox_dir)
                rst_msg += ("Now, you are at the root: " +
                            self._get_relative_cwd() + "\n\n")

            # Remove empty line
            output_lines = stdout.split("\n")
            while len(output_lines) and output_lines[-1] == "":
                output_lines.pop()

            if len(output_lines) == 0:
                return "Your bash code run smoothly without giving any outputs"

            os.chdir(output_lines[-1].strip().rstrip())
        except Exception as e:
            print(e)
            pdb.set_trace()

        return "\n".join(output_lines[:-1])

    def _run_command(self, cmd: list) -> str:
        """Inner function for self.run_command().
        Run a bash command in the dataset sandbox.

        The supported tools are:
        cd, ls, cat, echo, python, exit.

        Args:
        - cmd (list): a single command splitted into a list of arguments

        Returns:
        - str: the execution result of the given command. If any errors
        occurred, then just return the error message.
        """
        # Restrict command type
        if cmd[0] not in ["cd", "ls", "cat", "echo", "python", "exit"]:
            return "Error: You can only use cd, ls, cat, echo, python, and exit."

        # Test if echo outputs to a file
        if cmd[0] == "echo" and len(cmd) != 4:
            return "Warning: echo command without output file, ignored."

        if cmd[0] == "exit":
            return "Success: Bye!"
        else:
            # Test if the target file/dir is inside the sandbox
            target_dir = get_target_dir(cmd)
            if target_dir.startswith("Error:"):
                return target_dir

            if not target_dir.startswith(self.sandbox_dir):
                return (
                    f"Error: You cannot access files ({get_file_name(cmd)}) "
                    f"outside the repo! You are now at {self._get_relative_cwd()}"
                )

        # Run the command
        try:
            if cmd[0] == "cd":
                os.chdir(cmd[1])
                return "Success: Now at " + self._get_relative_cwd()
            else:
                result = subprocess.run(cmd, shell=True, capture_output=True)
                rstdout = result.stdout.decode('utf-8')
                rstderr = replace_absolute_with_relative(
                    result.stderr.decode('utf-8'), self.sandbox_dir)
                if cmd[0] == "ls":
                    return "Success: The result of ls is:\n" + rstdout
                elif cmd[0] == "cat":
                    return (f"Success: The content of {cmd[1]} is:\n" +
                            trunc_cat(cmd[1], rstdout))
                elif cmd[0] == "python":
                    if rstderr != "":
                        return "Error: " + rstderr
                    else:
                        return f"Success: The output of python is:\n{rstdout}"
        except Exception as e:
            return "Error: " + str(e)

    def get_changed_files(self) -> dict:
        """
        Return the name and content of changed files in the sandbox.

        Returns:
        - dict: key is relative file path, value is the content in bytes.
        """
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

        return {
            file: open(self.sandbox_dir + file, "rb").read()
            for file in changed_files
        }

    def _get_relative_cwd(self):
        "Return the relative path to the sandbox's root directory."
        return os.path.relpath(os.getcwd().replace('\\', '/'),
                               self.sandbox_dir) + "/"

    # def inject_and_run(self, llm_output: str) -> dict:
    #     """
    #     Given the output of LLM, retrieve the target source code and the
    #     injection snippet. Inject, run, and return the output.

    #     Args:
    #     - llm_output (str): The output of LLM, should be in natural language.
    #         - TARGET_FILE identifies the file to be appended with
    #         the injection snippet.
    #         - INJECTION_SNIPPET identifies the code to be injected.
    #         - COMMAND identifies the command to be run.

    #     Returns:
    #     - dict:
    #         - "stdout": The output of the injected code.
    #         - "stderr": The error message of the injected code.
    #         - "changed_files": A dict of changed files,
    #         key is relative file path,
    #         value is the content in bytes.
    #     """
    #     if "COMMAND" not in llm_output:
    #         print(colored("Command not found in response.", "yellow"))
    #         return {
    #             "stdout":
    #                 "",
    #             "stderr":
    #                 "",
    #             "changed_files": {},
    #             "information":
    #                 f"""You are now at the folder {self.relative_cwd()}"""
    #         }

    #     # Step 2: inject the new code into the sandbox
    #     while llm_output.find("TARGET_FILE:") != -1:
    #         target_file_start = llm_output.find("TARGET_FILE:") + len(
    #             "TARGET_FILE:")
    #         target_file_end = llm_output.find("\n", target_file_start)
    #         target_file = llm_output[target_file_start:target_file_end].strip()
    #         llm_output = llm_output[target_file_end:]

    #         injection_snippet_start = llm_output.find("INJECTION_SNIPPET:")
    #         injection_snippet_start = llm_output.find(
    #             "```python\n", injection_snippet_start) + len("```python\n")
    #         injection_snippet_end = llm_output.find("```",
    #                                                 injection_snippet_start)
    #         injection_snippet = llm_output[
    #             injection_snippet_start:injection_snippet_end]
    #         llm_output = llm_output[injection_snippet_end:]

    #         # Read existing content
    #         target_file_path = os.path.join(self.sandbox_dir, target_file)
    #         try:
    #             target_file_content = open(target_file_path, "r").read()
    #         except Exception as e:
    #             print(colored(f"Error {e}", "red"))
    #             target_file_content = ""

    #         # Write the new content
    #         target_file_content = target_file_content + "\n" + injection_snippet
    #         with open(target_file_path, "w") as f:
    #             f.write(target_file_content)

    #     print(colored("Injection done.", "green"))

    #     # extract commands
    #     bash_start = llm_output.find("COMMAND:")
    #     bash_start = llm_output.find("```bash\n", bash_start) + len("```bash\n")
    #     bash_end = llm_output.find("```", bash_start)
    #     bash = llm_output[bash_start:bash_end].strip()

    #     # run
    #     # os.chdir(self.sandbox_dir)
    #     # ignore all the warnings
    #     commands = shlex.split(bash)
    #     commands.insert(1, "-W ignore")
    #     # pdb.set_trace()
    #     result = subprocess.run(commands, capture_output=True)

    #     # find all changed files
    #     original_files = set(list_files(self.dataset_path))
    #     current_files = set(list_files(self.sandbox_dir))
    #     changed_files = list(current_files - original_files)

    #     common_files = current_files.intersection(original_files)

    #     for file in common_files:
    #         file = file.replace("\\", "/")

    #         original_file_path = self.dataset_path + file
    #         current_file_path = self.sandbox_dir + file

    #         original_file_content = open(original_file_path, "rb").read()
    #         current_file_content = open(current_file_path, "rb").read()

    #         if original_file_content != current_file_content:
    #             changed_files.append(file)

    #     print(colored("List of changed files:", "yellow"))
    #     print(changed_files)

    #     ret = {
    #         "stdout":
    #             result.stdout.decode('utf-8'),
    #         "stderr":
    #             replace_absolute_with_relative(result.stderr.decode('utf-8'),
    #                                            self.sandbox_dir),
    #         "changed_files": {
    #             file: open(self.sandbox_dir + file, "rb").read()
    #             for file in changed_files
    #         },
    #         "information":
    #             f"""You are now at the folder {self.relative_cwd()}"""
    #     }

    #     return ret
