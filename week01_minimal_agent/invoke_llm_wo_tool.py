import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ========== 第1轮对话 ==========
messages = [
    {'role': 'system', 'content': 'You are a daily working and learning summary assistant.'},
    {'role': 'user', 'content': 'i am learning the ai agent, make a plan template for me.'}
]

completion = client.chat.completions.create(
    model="qwen3.7-plus",  # 模型列表: https://help.aliyun.com/model-studio/getting-started/models
    messages=messages,
    n=1,  # 要求模型生成 3 个不同的回答
    temperature=0.9
)

reply_1 = completion.choices[0].message.content
print("【AI】", reply_1)

# ========== 第2轮对话 ==========
# 关键：把第1轮的对话历史追加到 messages 中
messages.append({'role': 'assistant', 'content': reply_1})  # 模型的回答
messages.append({'role': 'user', 'content': 'I am an AI engineer, so it can be more deep and challenge.'})  # 新的问题

completion = client.chat.completions.create(
    model="qwen3.7-plus",
    messages=messages
)

reply_2 = completion.choices[0].message.content
print("【AI】", reply_2)

