from gpt_api import get_llm

api = get_llm()

rst = api.reply("user",
                "How are you?",
                num_response=1,
                temperature=0.1,
                top_p=0.3,
                model="origin",
                max_tokens=100)

print(rst)

rst = api.reply("user",
                "How are you?",
                num_response=3,
                temperature=0.5,
                top_p=0.3,
                model="tuned",
                max_tokens=100)
print(rst)
