"""
Task 2（综合）：小红书爆款文案生成器
- System Prompt 负责"引导"（角色、风格）
- 代码负责"兜底"（字数、emoji 数量、钩子检查）
"""
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

SYSTEM_PROMPT = """你是一位顶级小红书爆款文案写手。
风格要求：
1. 第一行必须是钩子（制造好奇/冲突/利益点），不超过 20 字
2. 全文不超过 100 字
3. 至少使用 2 个 emoji
4. 语气口语化、有网感，多用"谁懂""真的绝了""必入"这类表达
5. 结尾加一个行动引导"""

TOPIC = "平价蓝牙耳机推荐"

def count_emoji(text: str) -> int:
    return len(re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", text))


def hard_check(text: str) -> dict:
    """代码层硬约束校验"""
    return {
        "字数<=100": len(text) <= 100,
        "emoji>=2": count_emoji(text) >= 2,
        "有钩子行": len(text.strip().split("\n")[0]) <= 20,
    }


def generate() -> str:
    r = client.chat.completions.create(
        model="glm-4.7-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"主题：{TOPIC}"},
        ],
        temperature=0.8,
        max_tokens=300,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return r.choices[0].message.content


print("=" * 60)
print(f"主题：{TOPIC}")
print("=" * 60)

for i in range(1, 4):
    text = generate()
    checks = hard_check(text)
    print(f"\n--- 版本 {i} ---")
    print(text)
    print(f"校验: {checks}  →  {'✅ 全部通过' if all(checks.values()) else '❌ 有未达标项'}")

print("\n" + "=" * 60)
print("注意：即使 Prompt 写了 100 字上限，代码仍会再校验一次。")
print("=" * 60)
