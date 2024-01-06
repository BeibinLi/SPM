import string

CHOICES = [str(i) for i in range(200)] + list(string.ascii_letters)
RESPONSE_TEMPLATE = " # Response:\n"


OVERRIDE_KEYS = ["model_name", "lora_r", "bf16", "fp16", "use_8bit", "use_4bit"]

DROPOUT_KEYS = ["resid_pdrop", "embd_pdrop", "attn_pdrop", "summary_first_dropout"]

DISABLE_DROPOUT_KWARGS = {k: 0 for k in DROPOUT_KEYS}



# exit should always be the last
SUPPORTED_CMDS = [
    "cd", "ls", "cat", "head", "tail", "echo", "python", "pip", "id", "exit"
]
FULL_CMDS = SUPPORTED_CMDS + [
    "pwd",
    "mkdir",
    "rmdir",
    "touch",
    "rm",
    "cp",
    "mv",
    "less",
    "grep",
    "find",
    "who",
    "w",
    "ps",
    "top",
    "kill",
    "tar",
    "chmod",
    "chown",
    "df",
    "du",
    "ifconfig",
    "ping",
    "netstat",
    "ssh",
    "scp",
]

# Common programming language suffixes
CODE_SUFFIXES = [
    ".py", ".c", ".cpp", ".cxx", ".cc", ".h", ".hpp", ".hxx", ".cs", ".java",
    ".go", ".ipynb"
]

# Common data file suffixes
DATA_SUFFIXES = [".csv", ".tsv", ".json", ".yaml", ".yml"]

# Common text file suffixes
TEXT_SUFFIXES = [".txt", ".md"]

# Executable file suffixes
EXEC_SUFFIXES = [".sh", ".bash", ".zsh"]

ALLOWED_FILE_SUFFIXES = CODE_SUFFIXES + DATA_SUFFIXES + TEXT_SUFFIXES + EXEC_SUFFIXES
