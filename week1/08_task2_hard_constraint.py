"""
Task 2（第 3 步）：软约束 vs 硬约束
同一个需求："回答不超过 20 字"
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

USER_MSG = "请写一篇 500 字的文章，详细介绍人工智能的发展历史。"
MAX_CHARS = 20


def ask(system_prompt: str) -> str:
    r = client.chat.completions.create(
        model="glm-4.7-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_MSG},
        ],
        temperature=0.3,
        max_tokens=300,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return r.choices[0].message.content


# ── 做法 1：只靠 System Prompt（软约束）──
soft = ask(f"你的回答绝对不能超过 {MAX_CHARS} 个中文字。")
print("=" * 60)
print("【做法 1】只靠 System Prompt")
print("=" * 60)
print(f"输出: {soft}")
print(f"字数: {len(soft)}  →  {'✅ 达标' if len(soft) <= MAX_CHARS else '❌ 超标'}")

# ── 做法 2：System Prompt + 代码兜底（硬约束）──
hard = ask(f"你的回答尽量精简。")
if len(hard) > MAX_CHARS:
    hard = hard[:MAX_CHARS]  # 代码强制截断
print()
print("=" * 60)
print("【做法 2】System Prompt + 代码截断")
print("=" * 60)
print(f"输出: {hard}")
print(f"字数: {len(hard)}  →  {'✅ 达标' if len(hard) <= MAX_CHARS else '❌ 超标'}")

print()
print("=" * 60)
print("结论：Prompt 负责'引导'，代码负责'兜底'。")
print("生产环境里，凡是不能出错的约束，都要在代码里再校验一次。")
print("=" * 60)
