"""
W3 第 4 课：多步工具链（Agent 的推理能力）

前一个工具的结果，决定下一个调什么 —— 这就是推理链。
同时加入 max_steps 安全阀，防止无限循环。
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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "在公司制度知识库中搜索文档",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
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
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

DOCS_DB = {
    "年假": "正式员工年假15天，入职满一年后方可享受。",
    "病假": "病假需提供医院证明，每年累计不超过30天。",
}


def execute_tool(name, args):
    if name == "search_docs":
        q = args.get("query", "")
        for k, v in DOCS_DB.items():
            if k in q:
                return v
        return f"ERROR: NOT_FOUND —— 未找到与「{q}」相关的制度。"
    if name == "calculate":
        try:
            return str(eval(args["expression"], {"__builtins__": {}}, {}))
        except Exception as e:
            return f"ERROR: 计算失败: {e}"
    return "ERROR: 未知工具"


def agent_loop(question, max_steps=6, verbose=True):
    messages = [{"role": "user", "content": question}]

    for step in range(1, max_steps + 1):        # ★ 安全阀：最多 max_steps 步
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model="glm-4.7-flash",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=500,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            if verbose:
                print(f"  [第{step}步] 模型不再调工具 → 给出最终回答")
            return msg.content, step

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = execute_tool(name, args)
            if verbose:
                print(f"  [第{step}步] 调 {name}{args} → 结果: {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # ★ 触发安全阀
    if verbose:
        print(f"  ⚠️ 达到最大步数 {max_steps}，强制停止（防止无限循环）")
    return "（已达最大步数，停止）", max_steps


print("=" * 60)
print("【多步工具链】前一步的结果决定下一步")
print("=" * 60)
q = "公司年假一共多少天？如果我请3天假，还剩几天？"
print(f"\n用户: {q}")
ans, steps = agent_loop(q)
print(f"\n最终回答: {ans}")
print(f"共执行 {steps} 步")

print("\n" + "=" * 60)
print("【安全阀测试】把 max_steps 设为 1，任务需要 2 步 → 应被强制停止")
print("=" * 60)
q2 = "公司年假一共多少天？如果我请3天假，还剩几天？"
print(f"\n用户: {q2}")
ans2, steps2 = agent_loop(q2, max_steps=1)
print(f"\n结果: {ans2}")
print(f"共执行 {steps2} 步（被安全阀拦截）")

print("\n" + "=" * 60)
print("【真正的 2 次工具调用】第二步必须查另一个文档")
print("=" * 60)
q3 = "公司年假和病假各有多少天？哪个更长？"
print(f"\n用户: {q3}")
ans3, steps3 = agent_loop(q3)
print(f"\n最终回答: {ans3}")
print(f"共执行 {steps3} 步（2 次工具调用 + 最终回答）")

print("\n" + "=" * 60)
print("关键：循环天然支持多步；max_steps 是防止模型失控的安全阀。")
print("=" * 60)
