import tempfile
import sys
import os
import io
import subprocess
import shlex

data_path = "../Coffee_Roasting_Dataset/"

def run_code(code: str) -> str:
    """
    Run the given code and return the output.
    
    Args:
    - code (str): A complete python code.

    Returns:
    - str: The output of the code.
    """
    
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False, suffix='.py') as temp:
        temp_file_name = temp.name
        temp.write(code)
    
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    with open(temp_file_name, 'r') as temp:
        exec(temp.read())

    captured_output = sys.stdout.getvalue()

    sys.stdout = original_stdout

    return captured_output

def inject_and_run(llm_output: str) -> str:
    """
    Given the output of LLM, retrieve the target source code and the injection snippet. Inject, run, and return the output.

    Args:
    - llm_output (str): The output of LLM, should be in natural language.

    Returns:
    - str: The output of the injected code.
    """

    roll_back_log = []

    while llm_output.find("TARGET_FILE:") != -1:
        target_file_start = llm_output.find("TARGET_FILE:") + len("TARGET_FILE:")
        target_file_end = llm_output.find("\n", target_file_start)
        target_file = llm_output[target_file_start:target_file_end].strip()
        llm_output = llm_output[target_file_end:]

        injection_snippet_start = llm_output.find("INJECTION_SNIPPET:")
        injection_snippet_start = llm_output.find("```python\n", injection_snippet_start) + len("```python\n")
        injection_snippet_end = llm_output.find("```", injection_snippet_start)
        injection_snippet = llm_output[injection_snippet_start:injection_snippet_end]
        llm_output = llm_output[injection_snippet_end:]

        target_file_path = data_path + target_file
        target_file_content = open(target_file_path, "r").read()

        roll_back_log.append((target_file_path, target_file_content))

        target_file_content = target_file_content + "\n" + injection_snippet

        with open(target_file_path, "w") as f:
            f.write(target_file_content)
    
    bash_start = llm_output.find("COMMAND:")
    bash_start = llm_output.find("```bash\n", bash_start) + len("```bash\n")
    bash_end = llm_output.find("```", bash_start)
    bash = llm_output[bash_start:bash_end].strip()

    original_cwd = os.getcwd()
    os.chdir(data_path)
    result = subprocess.run(shlex.split(bash), capture_output=True)
    
    os.chdir(original_cwd)
    # roll back
    # TODO: cannot detect all other files that are modified by the injected code, e.g., database, images
    for path, content in roll_back_log:
        with open(path, "w") as f:
            f.write(content)
    
    return result.stdout.decode('utf-8')

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
