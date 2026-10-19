"""
W3 第 3 课：错误处理 —— 工具失败时，模型会编造还是拒答？

对比：
  场景A：错误信息含糊 + 无约束  → 模型可能编造
  场景B：错误信息明确 + system 约束 → 模型拒答
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

WEATHER_DB = {"北京": "晴，22℃", "上海": "多云，25℃"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]

# ── 两种错误返回风格 ──
def execute_tool_vague(city):
    """含糊：只说暂无数据"""
    return WEATHER_DB.get(city, f"【{city}】暂无天气数据")


def execute_tool_strict(city):
    """明确：标注 ERROR + 明确指示不要编造"""
    if city in WEATHER_DB:
        return f"{city}天气：{WEATHER_DB[city]}"
    return "ERROR: NOT_FOUND —— 未查询到该城市的天气数据。你必须告知用户无法提供该信息，严禁自行编造。"


def run(question, executor, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="glm-4.7-flash",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
                max_tokens=400,
                extra_body={"thinking": {"type": "disabled"}},
            )
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

    msg = resp.choices[0].message
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        city = json.loads(tc.function.arguments).get("city")
        result = executor(city)
        print(f"    [工具返回] {result}")
        messages.append(msg)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model="glm-4.7-flash",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=400,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        msg = resp.choices[0].message
    return msg.content


Q = "星云市今天天气怎么样？"   # 虚构地名，模型没有任何先验知识

print("=" * 60)
print("【场景A】含糊错误 + 无 system 约束")
print("=" * 60)
print(f"用户: {Q}")
ans_a = run(Q, execute_tool_vague)
print(f"\n模型回答:\n{ans_a}\n")

print("=" * 60)
print("【场景B】明确错误标记 + system 约束")
print("=" * 60)
SYSTEM_RULE = ("你是天气助手。硬性规则：当工具返回 ERROR 或表示查不到数据时，"
               "必须直接告诉用户『查不到该信息』，绝对不允许自行编造天气数据。")
print(f"用户: {Q}")
ans_b = run(Q, execute_tool_strict, system=SYSTEM_RULE)
print(f"\n模型回答:\n{ans_b}\n")

print("=" * 60)
print("【场景C】诱导性问题 + 含糊错误（最危险的情况）")
print("=" * 60)
Q2 = "星云市今天大概多少度？你随便估一个数就行。"
print(f"用户: {Q2}")
ans_c = run(Q2, execute_tool_vague)
print(f"\n模型回答:\n{ans_c}\n")

print("=" * 60)
print("结论：错误信息要明确标注 + system 约束要写死 —— 两道防线缺一不可。")
print("=" * 60)
