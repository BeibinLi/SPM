import tempfile
import sys
import io


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

