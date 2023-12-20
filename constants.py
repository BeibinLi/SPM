import string

CHOICES = [str(i) for i in range(100)] + list(string.ascii_letters)
RESPONSE_TEMPLATE = " # Response:\n"


OVERRIDE_KEYS = ["model_name", "lora_r", "bf16", "fp16", "use_8bit", "use_4bit"]

DROPOUT_KEYS = ["resid_pdrop", "embd_pdrop", "attn_pdrop", "summary_first_dropout"]

DISABLE_DROPOUT_KWARGS = {k: 0 for k in DROPOUT_KEYS}
