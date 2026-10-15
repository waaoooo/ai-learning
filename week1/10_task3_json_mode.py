"""
Task 3（第 1 步）：让 LLM 输出 JSON —— 直接要 vs 用 JSON Mode
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

USER_MSG = "提取这句话里的信息：张三，25 岁，会 Python 和 JavaScript，目前还在上大学。"


def try_parse(label, raw):
    print(f"\n【{label}】")
    print(f"原始输出:\n{raw}")
    try:
        data = json.loads(raw)
        print(f"✅ 解析成功: {data}")
    except json.JSONDecodeError as e:
        print(f"❌ 解析失败: {e}")


# ── 做法 1：只在 Prompt 里说"输出 JSON" ──
r1 = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[
        {"role": "system", "content": "你是一个信息提取助手，请严格按照 JSON 格式输出。"},
        {"role": "user", "content": USER_MSG},
    ],
    temperature=0.1,
    max_tokens=300,
    extra_body={"thinking": {"type": "disabled"}},
)
try_parse("做法 1：只靠 Prompt", r1.choices[0].message.content)

# ── 做法 2：启用 JSON Mode ──
r2 = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[
        {"role": "system", "content": "你是一个信息提取助手。"},
        {"role": "user", "content": USER_MSG},
    ],
    temperature=0.1,
    max_tokens=300,
    response_format={"type": "json_object"},  # 关键
    extra_body={"thinking": {"type": "disabled"}},
)
try_parse("做法 2：JSON Mode", r2.choices[0].message.content)
