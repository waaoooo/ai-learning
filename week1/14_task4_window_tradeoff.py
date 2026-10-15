"""
Task 4（第 3 步）：滑动窗口的代价 —— 只有早期历史才答得出的问题
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

HISTORY = []
for i in range(1, 21):
    HISTORY.append({"role": "user", "content": f"第 {i} 轮：我们来聊聊主题 {i}，请简单回应一句。"})
    HISTORY.append({"role": "assistant", "content": f"好的，关于主题 {i} 我说一句：这是第 {i} 轮。"})

# 只有第 1 轮才包含答案的问题
QUESTION = "第 1 轮的主题编号是多少？"


def ask(history):
    r = client.chat.completions.create(
        model="glm-4.7-flash",
        messages=history + [{"role": "user", "content": QUESTION}],
        temperature=0.1,
        max_tokens=100,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return r


for label, hist in [
    ("全留（20 轮）", HISTORY),
    ("滑动窗口（最近 5 轮）", HISTORY[-10:]),
]:
    r = ask(hist)
    print("=" * 60)
    print(f"【{label}】")
    print("=" * 60)
    print(f"  回答: {r.choices[0].message.content}")
    print(f"  total_tokens: {r.usage.total_tokens}")
    print()

print("=" * 60)
print("结论：滑动窗口省钱，但会丢失早期信息。")
print("生产环境常用补充：把重要信息（用户偏好、关键结论）单独存起来，")
print("每次作为 system 或摘要注入，而不是依赖原始对话历史。")
print("=" * 60)
