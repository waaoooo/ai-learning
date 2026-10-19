"""
W3 第 1 课：让模型【自主选择】工具（tool_choice="auto"）

对比 W1：W1 用 tool_choice="required" 强制调用；
        这里给 3 个工具，让模型自己判断该用哪个。
"""
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# ── 工具定义（3 个候选工具）──
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如：北京"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如：1+2*3"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "在公司知识库中搜索制度文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    },
]

# (问题, 期望模型选的工具)
QUESTIONS = [
    ("北京今天天气怎么样？", "get_weather"),
    ("1234 乘以 5678 等于多少？", "calculate"),
    ("公司的年假制度是怎样的？", "search_docs"),
    ("你好呀", None),   # 不需要工具的情况
]


def ask(question):
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model="glm-4.7-flash",
                messages=[{"role": "user", "content": question}],
                tools=TOOLS,
                tool_choice="auto",          # ★ 关键：让模型自己选
                temperature=0,
                max_tokens=500,
                extra_body={"thinking": {"type": "disabled"}},
            ).choices[0].message
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


print("=" * 60)
print("工具自主选择测试（tool_choice='auto'）")
print("=" * 60)

for q, expect in QUESTIONS:
    msg = ask(q)
    calls = msg.tool_calls

    if not calls:
        print(f"\n问: {q}")
        print(f"  模型选择: 不调用任何工具，直接回答")
        print(f"  回复: {msg.content[:60]}")
        ok = (expect is None)
        print(f"  {'✅ 正确判断：这个问题不需要工具' if ok else '❌ 预期应调用 ' + str(expect)}")
        continue

    tc = calls[0]
    fn = tc.function.name
    args = json.loads(tc.function.arguments)
    ok = (fn == expect)
    print(f"\n问: {q}")
    print(f"  模型选择: {fn}")
    print(f"  传入参数: {args}")
    print(f"  {'✅' if ok else '❌'} 预期: {expect}")

print("\n" + "=" * 60)
print("结论：模型根据问题语义，自己判断该用哪个工具 —— 这就是 Agent 的起点。")
print("=" * 60)
