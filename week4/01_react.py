"""
W4 第 1 课：ReAct 模式（Reasoning + Acting）

让模型显式输出"思考 → 行动 → 观察"，把推理过程写出来。
对比 W3：W3 的推理藏在 tool_calls 里，看不到；ReAct 把它显式化。
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

# ── 工具（真实执行）──
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


# ── ReAct system prompt：规定"先想后做"的格式 ──
REACT_SYSTEM = """你是一个严谨的助手。解决问题时必须严格按以下格式，一步一步来：

Thought: 你的推理过程（这一步要做什么、为什么）
Action: 工具名
Action Input: 参数（JSON 格式）

当你已经能回答时，输出：
Thought: 我现在可以回答了
Final Answer: 你的最终回答

可用工具：
- get_weather(city)：查询城市天气
- search_docs(query)：搜索公司制度

规则：每次只输出一步（一个 Thought + 一个 Action），不要一次输出多步。"""


def react(question, max_steps=6):
    messages = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model="glm-4.7-flash",
                    messages=messages,
                    temperature=0,
                    max_tokens=600,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

        out = resp.choices[0].message.content
        print(f"\n── 第 {step} 步 ──")
        print(out.strip())
        messages.append({"role": "assistant", "content": out})

        # 得到最终答案 → 结束
        if "Final Answer:" in out:
            return out.split("Final Answer:")[-1].strip()

        # 解析 Action + Action Input
        m_action = re.search(r"Action:\s*(\w+)", out)
        m_input = re.search(r"Action Input:\s*(\{.*\})", out)
        if not m_action or not m_input:
            messages.append({"role": "user", "content": "Observation: 格式错误，请按 Thought/Action/Action Input 格式输出。"})
            continue

        name = m_action.group(1)
        try:
            args = json.loads(m_input.group(1))
        except json.JSONDecodeError:
            args = {}

        result = execute(name, args)
        print(f"   ↳ 代码执行 {name}{args} → {result}")
        # 把观察结果喂回去
        messages.append({"role": "user", "content": f"Observation: {result}"})

    return "（达到最大步数，停止）"


print("=" * 60)
print("ReAct 模式：让模型先想后做")
print("=" * 60)
q = "公司年假有多少天？如果我请3天假，还剩几天？"
print(f"\n用户: {q}\n")
ans = react(q)
print("\n" + "=" * 60)
print(f"最终答案: {ans}")
print("=" * 60)
