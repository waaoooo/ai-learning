"""
Task 2（第 1 步）：同一个问题，不同 System Prompt → 不同回答
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

QUESTION = "什么是机器学习？"


def ask(system_prompt: str) -> str:
    msgs = [{"role": "user", "content": QUESTION}]
    if system_prompt:
        msgs.insert(0, {"role": "system", "content": system_prompt})
    r = client.chat.completions.create(
        model="glm-4.7-flash",
        messages=msgs,
        temperature=0.3,
        max_tokens=200,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return r.choices[0].message.content


cases = [
    ("无 System Prompt", ""),
    ("角色：给 5 岁小孩讲", "你是一个幼儿园老师，要用 5 岁小孩能听懂的话讲解，多用比喻，不超过 80 字。"),
    ("角色：严谨教授", "你是一位大学教授，回答必须严谨、使用学术语言，先给定义再给例子，不超过 80 字。"),
]

for label, sp in cases:
    print("=" * 60)
    print(f"【{label}】")
    print("=" * 60)
    print(ask(sp))
    print()
