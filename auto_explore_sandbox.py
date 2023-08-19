import tempfile
import os
import subprocess
import shutil
import random
import string
from hashlib import sha256
from termcolor import colored
from utils import (list_files, get_target_dirs, hide_root, trunc_text,
                   get_file_names, SUPPORTED_CMDS)

SAFE_MESSAGE = "SAFE"


class AutoExploreSandbox:

    def __init__(
        self,
        dataset_path: str,
        password: str,
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
        # Upon deletion, clean up the temporary directory
        # If it is windows, run 'rmdir'
        # otherwise, run rm -rf
        if os.name == "nt":
            os.system('rmdir /S /Q "{}"'.format(self.sandbox_dir))
        else:
            os.system('rm -rf "{}"'.format(self.sandbox_dir))

    def _hash_password(self, password: str) -> str:
        return sha256((password + self._sandbox_id).encode("utf-8")).hexdigest()

    def safety_check(self, cmd: list, password: str) -> str:
        """
        Return "SAFE" iff the cmd is safe to run.
        Otherwise, return error message.

        Args:
        - cmd (list): a single command splitted into a list of arguments.
        - password (str): the password for identity verification.

        Returns:
        """
        # First check if password is correct
        if self._hash_password(password) != self._hashed_password:
            return "Error: Wrong password!"

        # Restrict command type
        if cmd[0] == "exit":
            raise NotImplementedError(
                "exit should be handled outside of run_command().")
        if cmd[0] not in SUPPORTED_CMDS:
            return f"Error: You can only use {', '.join(SUPPORTED_CMDS[:-1])}."

        # Test if the target file/dir is inside the sandbox
        target_dirs = get_target_dirs(cmd)
        for target_dir in target_dirs:
            if "Error" in target_dir:
                return target_dir
            if not target_dir.startswith(self.sandbox_dir):
                return (
                    f"Error: You cannot access file {target_dir} "
                    f"outside the repo! You are now at {self._get_relative_cwd()}"
                )

        # Check if the target file is private
        files = get_file_names(cmd)
        for file in files:
            if file in self.private_files:
                return f"Error: You cannot access a private file {file}!"

        return SAFE_MESSAGE

    def run_command(self, cmd: list, password: str) -> str:
        """Wrapper function for self._run_command().
        Run a bash command in the dataset sandbox.

        The supported tools are:
        "cd", "ls", "cat", "head", "tail", "echo", "python", "pip"
        "exit" is handled outside of this function.

        Args:
        - cmd (list): a single command splitted into a list of arguments.
        - password (str): the password for identity verification.

        Returns:
        - str: the execution result of the given command. If any errors
        occurred, then just return the error message.
        """
        # Restore to the checkpointed cwd
        _cwd = os.getcwd()
        os.chdir(self.cwd)

        safety_check_result = self.safety_check(cmd, password)

        if safety_check_result != SAFE_MESSAGE:
            ret = safety_check_result
        else:
            ret = self._run_command(cmd)

        # Checkpoint cwd
        self.cwd = os.getcwd().replace("\\", "/") + "/"
        os.chdir(_cwd)

        return ret

    def _run_command(self, cmd: list) -> str:
        """Inner function for self.run_command().
        Run a bash command in the dataset sandbox.

        The supported tools are:
        "cd", "ls", "cat", "head", "tail", "echo", "python", "pip".
        "exit" is handled outside of this function.

        Args:
        - cmd (list): a single command splitted into a list of arguments.

        Returns:
        - str: the execution result of the given command. If any errors
        occurred, then just return the error message.
        """

        # Check if echo outputs to a file
        if cmd[0] == "echo" and len(cmd) == 3:
            return "Warning: echo command without output file, ignored."

        # Run the command
        try:
            if cmd[0] == "cd":
                # cd cannot be handled by subprocess
                os.chdir(cmd[1])
                return "Success: Now at " + self._get_relative_cwd()
            else:
                result = subprocess.run(' '.join(cmd),
                                        shell=True,
                                        capture_output=True)
                return self.respond_cmd(cmd, result)
        except Exception as e:
            return "Error: " + str(e)

    def respond_cmd(self, cmd: list, result) -> str:
        """
        Generate the response for the result of a command.

        Args:
        - cmd (list): a single command splitted into a list of arguments.
        - result (subprocess.CompletedProcess): the result of the command.

        Returns:
        - str: the response for the result of the command.
        """
        rstdout = result.stdout.decode('utf-8')
        rstderr = hide_root(result.stderr.decode('utf-8'), self.sandbox_dir)

        if cmd[0] == "ls":
            return "Success: The result of ls is:\n" + rstdout
        elif cmd[0] in ["cat", "head", "tail"]:
            fn = get_file_names(cmd)[0]
            return (f"Success: The content of {fn} is:\n" +
                    trunc_text(fn, rstdout))
        elif cmd[0] == "echo":
            return f"Success: echoed to {cmd[-1]}"
        elif cmd[0] == "python":
            if rstderr != "":
                return f"Error: {rstderr}"
            else:
                return f"Success: The output of python is:\n{rstdout}"
        elif cmd[0] == "pip":
            if rstderr != "":
                return f"Error: {rstderr}"
            else:
                return "Success: pip succeeded"
        else:
            raise NotImplementedError(f"Does not support command: {cmd[0]}")

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
