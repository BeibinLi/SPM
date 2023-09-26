import os

raw_data_path = "../Coffee_Roasting_Dataset/data/"
uri_raw_data_path = raw_data_path + "URI/"
self_instruct_raw_data_path = (
    raw_data_path +
    "self-instruct/finetuning/self_instruct_221203/gpt3_finetuning_data.jsonl")

prompt_template_path = "data_gen/prompt_templates/"
uri_attr_to_lang_keyw_prompt_path = (prompt_template_path +
                                     "uri/uri_attribute_to_language_keyword.md")
uri_df_prompt_path = prompt_template_path + "uri/df/"
reading_comp_q_prompt_path = (
    prompt_template_path +
    "intermediate/reading_comprehension_with_question.md")

data_path = os.path.abspath(os.path.expanduser("data"))
chatlog_output_path = os.path.join(data_path, "chatlogs")
prompt_output_path = os.path.join(data_path, "prompts")
pretrain_data_path = os.path.join(data_path, "pretrain")
finetune_data_path = os.path.join(data_path, "finetune")
self_instruct_data_path = os.path.join(data_path, "self-instruct")
pretrain_raw_data_path = os.path.join(data_path, "pretrain_raw")
