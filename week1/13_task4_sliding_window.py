"""
Task 4（第 2 步）：滑动窗口 —— 用真实 token 数对比"全留" vs "只留最近 N 轮"
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# 模拟一段 20 轮的对话历史
HISTORY = []
for i in range(1, 21):
    HISTORY.append({"role": "user", "content": f"第 {i} 轮：我们来聊聊主题 {i}，请简单回应一句。"})
    HISTORY.append({"role": "assistant", "content": f"好的，关于主题 {i} 我说一句：这是第 {i} 轮。"})

QUESTION = "我们一共聊了多少轮？"


def build_messages(history: list, max_turns: int = None) -> list:
    """max_turns=None 表示全留；数字表示只留最近 N 轮"""
    if max_turns is not None:
        history = history[-(max_turns * 2):]  # 每轮 = user + assistant
    return history + [{"role": "user", "content": QUESTION}]


def call(messages):
    r = client.chat.completions.create(
        model="glm-4.7-flash",
        messages=messages,
        temperature=0.1,
        max_tokens=100,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return r


for label, max_turns in [("全留（20 轮）", None), ("滑动窗口（最近 5 轮）", 5)]:
    msgs = build_messages(HISTORY, max_turns)
    r = call(msgs)
    print("=" * 60)
    print(f"【{label}】  发送消息数: {len(msgs)}")
    print("=" * 60)
    print(f"  回答: {r.choices[0].message.content}")
    print(f"  prompt_tokens: {r.usage.prompt_tokens}")
    print(f"  completion_tokens: {r.usage.completion_tokens}")
    print(f"  total_tokens: {r.usage.total_tokens}")
    print()
