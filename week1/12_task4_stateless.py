"""
Task 4（第 1 步）：LLM 无状态 —— 不传历史，它就失忆
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)


def ask(messages):
    r = client.chat.completions.create(
        model="glm-4.7-flash",
        messages=messages,
        temperature=0.3,
        max_tokens=100,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return r.choices[0].message.content


print("=" * 60)
print("【场景】用户先告诉模型自己的名字，再问“我叫什么”")
print("=" * 60)

# 第 1 次调用：告诉模型名字（模拟上一轮）
print("\n>>> 上一轮：用户说“我叫小明”")
ask([{"role": "user", "content": "我叫小明"}])

# 第 2 次调用：不传历史，直接问
print(">>> 这一轮：不传任何历史，直接问")
answer = ask([{"role": "user", "content": "我叫什么名字？"}])
print(f"模型回答: {answer}")
print("↑ 它根本不知道，因为这次调用里没有“我叫小明”")

# 第 3 次调用：把历史带上
print("\n>>> 这一轮：把上一轮的消息一起传进去")
answer = ask([
    {"role": "user", "content": "我叫小明"},
    {"role": "user", "content": "我叫什么名字？"},
])
print(f"模型回答: {answer}")
print("↑ 这次它记得了，因为历史是你塞进去的")
