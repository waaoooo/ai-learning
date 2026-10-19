"""
W4 第 2 课：任务规划（Plan-and-Execute）

阶段1：让模型把复杂任务拆解成步骤（结构化 JSON）
阶段2：按计划逐步执行，每步可调工具
"""
import os
import re
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

DOCS_DB = {"年假": "正式员工年假15天，入职满一年后方可享受。",
           "病假": "病假需提供医院证明，每年累计不超过30天。"}

def tool_get_weather(city: str) -> str:
    db = {"北京": "晴，22℃", "上海": "多云，25℃"}
    return f"{city}天气：{db[city]}" if city in db else f"ERROR: 未查到「{city}」"

def tool_search_docs(query: str) -> str:
    for k, v in DOCS_DB.items():
        if k in query:
            return v
    return f"ERROR: 未找到「{query}」相关制度"

def execute(name, args):
    if name == "get_weather":
        return tool_get_weather(args.get("city", ""))
    if name == "search_docs":
        return tool_search_docs(args.get("query", ""))
    return "ERROR: 未知工具"


TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "查询城市天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_docs", "description": "搜索公司制度",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

PLAN_SYSTEM = """你是一个任务规划器。把用户的任务拆解成清晰、可执行的具体步骤。

你拥有以下工具（规划步骤时必须明确说明调用哪个工具）：
- get_weather(city)：查询城市天气
- search_docs(query)：搜索公司制度

用 JSON 输出，格式：{"steps": ["步骤1", "步骤2", ...]}
要求：步骤要具体、有序、不遗漏关键动作；涉及数据查询的步骤必须指明用哪个工具。只输出 JSON，不要其他文字。"""


def call(messages, json_mode=False, tools=None):
    for attempt in range(4):
        try:
            kwargs = dict(model="glm-4.7-flash", messages=messages, temperature=0,
                          max_tokens=600, extra_body={"thinking": {"type": "disabled"}})
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if tools:                       # ★ 执行阶段必须把工具传给模型
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return client.chat.completions.create(**kwargs)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


# ══════════════════════════════════════════════════════════
# 阶段 1：规划
# ══════════════════════════════════════════════════════════
def make_plan(task: str) -> list:
    resp = call([{"role": "system", "content": PLAN_SYSTEM},
                 {"role": "user", "content": task}], json_mode=True)
    return json.loads(resp.choices[0].message.content)["steps"]


# ══════════════════════════════════════════════════════════
# 阶段 2：执行单步（可调用工具）
# ══════════════════════════════════════════════════════════
def run_step(step: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "你是执行助手，只能根据工具返回的信息回答，不要编造。"},
        {"role": "user", "content": f"已知信息：{context}\n\n请完成这一步：{step}"},
    ]
    for _ in range(4):   # 单步内允许最多 2 次工具调用
        resp = call(messages, tools=TOOLS)
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return msg.content
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = execute(name, args)
            print(f"      ↳ 调 {name}{args} → {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "（本步未完成）"


# ══════════════════════════════════════════════════════════
# 演示
# ══════════════════════════════════════════════════════════
TASK = ("帮我查一下北京今天的天气，再查公司年假天数，"
        "告诉我如果请3天假还剩几天，最后给一句出行建议。")

print("=" * 60)
print("阶段 1：规划（把任务拆成步骤）")
print("=" * 60)
print(f"\n任务: {TASK}\n")
steps = make_plan(TASK)
for i, s in enumerate(steps, 1):
    print(f"  步骤{i}: {s}")

print("\n" + "=" * 60)
print("阶段 2：按计划逐步执行")
print("=" * 60)

context = ""
results = []
for i, s in enumerate(steps, 1):
    print(f"\n  【执行步骤{i}】{s}")
    r = run_step(s, context)
    results.append(r)
    context += f"\n- {s} → {r}"
    print(f"      结果: {r}")

print("\n" + "=" * 60)
print("最终汇总")
print("=" * 60)
final = call([{"role": "system", "content": "你是助手，根据以下步骤结果，给用户一个简洁的完整回答。"},
              {"role": "user", "content": f"任务：{TASK}\n\n各步骤结果：{context}"}])
print(final.choices[0].message.content)

print("\n" + "=" * 60)
print("关键：先规划（不遗漏）→ 再执行（每步可验证）→ 最后汇总。")
print("=" * 60)
