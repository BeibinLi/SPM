from gpt_api import get_llm

api = get_llm()

rst = api.reply("user",
                "Hello",
                num_response=1,
                temperature=0.1,
                top_p=0.3,
                model="origin")

print(rst)

rst = api.reply("user",
                "Hello",
                num_response=3,
                temperature=0.5,
                top_p=0.3,
                model="tuned")
print(rst)
