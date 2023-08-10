import tempfile
import sys
import os
import io
import subprocess
import shlex
import shutil
from termcolor import colored
from utils import list_files

dataset_path = os.path.abspath("../Coffee_Roasting_Dataset/") + "/"
# use absolute path to avoid problems with cwd

def run_code(code: str) -> str:
    """
    Run the given code and return the output.

    Args:
    - code (str): A complete python code.

    Returns:
    - str: The output of the code.
    """

    with tempfile.NamedTemporaryFile(mode='w+t', delete=False,
                                     suffix='.py') as temp:
        temp_file_name = temp.name
        temp.write(code)

    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    with open(temp_file_name, 'r') as temp:
        exec(temp.read())

    captured_output = sys.stdout.getvalue()

    sys.stdout = original_stdout

    return captured_output


def inject_and_run(llm_output: str) -> dict:
    """
    Given the output of LLM, retrieve the target source code and the injection
    snippet. Inject, run, and return the output.

    Args:
    - llm_output (str): The output of LLM, should be in natural language.

    Returns:
    - dict:
        - "stdout": The output of the injected code.
        - "stderr": The error message of the injected code.
        - "changed_files": A dict of changed files, key is relative file path, value is the content in bytes.
    """

    # copy dataset to a temporary directory
    temp_dir = tempfile.mkdtemp()
    shutil.copytree(dataset_path,
                    f"{temp_dir}/{dataset_path.split('/')[-1]}",
                    dirs_exist_ok=True)
    print(colored(f"Data copied to temporary directory: {temp_dir}", "green"))
    temp_dir += "/"

    # inject
    while llm_output.find("TARGET_FILE:") != -1:
        target_file_start = llm_output.find("TARGET_FILE:") + len(
            "TARGET_FILE:")
        target_file_end = llm_output.find("\n", target_file_start)
        target_file = llm_output[target_file_start:target_file_end].strip()
        llm_output = llm_output[target_file_end:]

        injection_snippet_start = llm_output.find("INJECTION_SNIPPET:")
        injection_snippet_start = llm_output.find(
            "```python\n", injection_snippet_start) + len("```python\n")
        injection_snippet_end = llm_output.find("```", injection_snippet_start)
        injection_snippet = llm_output[
            injection_snippet_start:injection_snippet_end]
        llm_output = llm_output[injection_snippet_end:]

        target_file_path = temp_dir + target_file
        target_file_content = open(target_file_path, "r").read()

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
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    # make sure all the code is runnable in the root of dataset directory
    result = subprocess.run(shlex.split(bash), capture_output=True)

    # find all changed files
    original_files = set(list_files(dataset_path))
    current_files = set(list_files(temp_dir))
    changed_files = list(current_files - original_files)

    common_files = current_files.intersection(original_files)

    for file in common_files:
        original_file_path = dataset_path + file
        current_file_path = temp_dir + file

        original_file_content = open(original_file_path, "rb").read()
        current_file_content = open(current_file_path, "rb").read()

        if original_file_content != current_file_content:
            changed_files.append(file)

    print(colored("List of changed files:", "yellow"))
    print(changed_files)

    ret = {
        "stdout": result.stdout.decode('utf-8'),
        "stderr": result.stderr.decode('utf-8'),
        "changed_files": {
            file: open(temp_dir + file, "rb").read() for file in changed_files
        }
    }

    # restore the original cwd
    os.chdir(original_cwd)

    # clean up the temporary directory
    shutil.rmtree(temp_dir)

    return ret


if __name__ == "__main__":
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
    print(inject_and_run(llm_output))
