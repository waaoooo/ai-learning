"""
W3 第 2 课：工具执行循环（Agent 的心脏）

模型负责"决定调什么"，代码负责"真的执行"，再把结果喂回给模型。
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
            "name": "get_weather",
            "description": "查询指定城市的天气",
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
]

WEATHER_DB = {
    "北京": "晴，22℃，东风3级，适合出行",
    "上海": "多云，25℃，东南风2级",
    "深圳": "雷阵雨，28℃，注意带伞",
}


# ── ② 代码执行工具（注意：生产环境不要用 eval 处理不可信输入）──
def execute_tool(name, args):
    if name == "calculate":
        try:
            return str(eval(args["expression"], {"__builtins__": {}}, {}))
        except Exception as e:
            return f"计算出错: {e}"
    if name == "get_weather":
        return WEATHER_DB.get(args["city"], f"【{args['city']}】暂无天气数据")
    return "未知工具"


def agent_loop(question, verbose=True):
    messages = [{"role": "user", "content": question}]
    steps = 0

    while True:
        steps += 1
        for attempt in range(4):          # 免费模型限流频繁，循环里必须有重试
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
                print(f"        (限流，{2 ** attempt}s 后重试)")
                time.sleep(2 ** attempt)
        msg = resp.choices[0].message
        messages.append(msg)          # 把模型的回复（含 tool_call）加入历史

        # 模型没要求调工具 → 给出最终回答，循环结束
        if not msg.tool_calls:
            if verbose:
                print(f"  [第{steps}步] 模型给出最终回答")
            return msg.content

        # ② 代码执行工具
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = execute_tool(name, args)
            if verbose:
                print(f"  [第{steps}步] 模型要调: {name}{args}")
                print(f"        代码执行 → 结果: {result}")
            # ③ 把结果喂回给模型
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


print("=" * 60)
print("【场景1】需要真实数据 → 触发工具循环")
print("=" * 60)
q1 = "北京今天天气怎么样？适合出门吗？"
print(f"\n用户: {q1}")
ans1 = agent_loop(q1)
print(f"\n最终回答: {ans1}")

print("\n" + "=" * 60)
print("【场景2】需要计算 → 触发工具循环")
print("=" * 60)
q2 = "1234 乘以 5678 等于多少？"
print(f"\n用户: {q2}")
ans2 = agent_loop(q2)
print(f"\n最终回答: {ans2}")

print("\n" + "=" * 60)
print("【场景3】不需要工具 → 循环一步结束")
print("=" * 60)
q3 = "给我讲个程序员笑话"
print(f"\n用户: {q3}")
ans3 = agent_loop(q3)
print(f"\n最终回答: {ans3}")

print("\n" + "=" * 60)
print("关键：模型只负责'决定'，代码负责'执行' —— 这就是 Agent。")
print("=" * 60)
