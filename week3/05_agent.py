"""
W3 第 5 课：综合 Agent —— 把所有能力缝起来

架构原则：确定性的事交给代码，不确定性的事交给 LLM。
  代码负责：流程编排（要不要调工具/循环几次/失败怎么办/记忆保留几轮）
  LLM 负责：内容决策（调哪个工具/传什么参数/怎么回答）
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

DOCS_DB = {
    "年假": "正式员工年假15天，入职满一年后方可享受。",
    "病假": "病假需提供医院证明，每年累计不超过30天。",
    "加班": "工作日加班1.5倍工资，周末2倍，法定节假日3倍。",
}


# ══════════════════════════════════════════════════════════
# ① 工具注册（可扩展：加工具只需写一个函数 + 一个 schema）
# ══════════════════════════════════════════════════════════
def tool_get_weather(city: str) -> str:
    """查询城市天气"""
    db = {"北京": "晴，22℃", "上海": "多云，25℃", "深圳": "雷阵雨，28℃"}
    if city in db:
        return f"{city}天气：{db[city]}"
    return f"ERROR: NOT_FOUND —— 未查询到「{city}」的天气数据，请告知用户无法提供。"


def tool_calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"ERROR: 计算失败: {e}"


def tool_search_docs(query: str) -> str:
    """搜索公司制度文档"""
    for k, v in DOCS_DB.items():
        if k in query:
            return v
    return f"ERROR: NOT_FOUND —— 未找到与「{query}」相关的制度。"


TOOLS = [
    {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        "func": tool_get_weather,
    },
    {
        "name": "calculate",
        "description": "计算数学表达式",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        "func": tool_calculate,
    },
    {
        "name": "search_docs",
        "description": "在公司制度知识库中搜索文档",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "func": tool_search_docs,
    },
]


# ══════════════════════════════════════════════════════════
# ② Agent：代码做编排，LLM 做决策
# ══════════════════════════════════════════════════════════
class Agent:
    def __init__(self, system: str, tools: list, model="glm-4.7-flash",
                 max_steps=6, history_limit=10):
        self.system = system
        self.tools = {t["name"]: t for t in tools}
        self.model = model
        self.max_steps = max_steps          # 安全阀：防无限循环
        self.history_limit = history_limit  # 记忆：保留最近 N 轮
        self.history = []

    def _llm(self, messages):
        """调 LLM（带限流重试）"""
        for attempt in range(4):
            try:
                return client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._schemas(),
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=600,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def _schemas(self):
        return [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in self.tools.values()]

    def _compose(self):
        """编排：system + 最近 N 轮记忆"""
        return ([{"role": "system", "content": self.system}]
                + self.history[-self.history_limit:])

    def _execute(self, name, args):
        """执行工具，出错不崩溃（错误也喂回给模型）"""
        fn = self.tools.get(name, {}).get("func")
        if not fn:
            return f"ERROR: 未知工具 {name}"
        try:
            return fn(**args)
        except Exception as e:
            return f"ERROR: 工具执行异常: {e}"

    def chat(self, user_input: str, verbose=True) -> str:
        """多轮对话入口"""
        self.history.append({"role": "user", "content": user_input})
        messages = self._compose()

        for step in range(1, self.max_steps + 1):     # ★ 编排：循环控制
            resp = self._llm(messages)
            msg = resp.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:                    # ★ 编排：终止条件
                self.history.append({"role": "assistant", "content": msg.content})
                return msg.content

            for tc in msg.tool_calls:                 # ★ 编排：执行工具
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = self._execute(name, args)
                if verbose:
                    print(f"    [第{step}步] {name}{args} → {result}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return "（已达最大步数，停止）"


# ══════════════════════════════════════════════════════════
# ③ 使用
# ══════════════════════════════════════════════════════════
SYSTEM = ("你是公司制度助手。硬性规则：工具返回 ERROR 或查不到数据时，"
          "必须直接告知用户查不到，严禁编造。")

agent = Agent(SYSTEM, TOOLS, max_steps=6, history_limit=10)

print("=" * 60)
print("【场景1】单工具调用")
print("=" * 60)
print("\n用户: 北京今天天气怎么样？")
print(f"助手: {agent.chat('北京今天天气怎么样？')}\n")

print("=" * 60)
print("【场景2】多步工具链")
print("=" * 60)
print("\n用户: 公司年假和病假各有多少天？哪个更长？")
print(f"助手: {agent.chat('公司年假和病假各有多少天？哪个更长？')}\n")

print("=" * 60)
print("【场景3】错误处理（查不到时不编造）")
print("=" * 60)
print("\n用户: 星云市天气怎么样？")
print(f"助手: {agent.chat('星云市天气怎么样？')}\n")

print("=" * 60)
print("【场景4】多轮记忆（还记得前面的内容吗？）")
print("=" * 60)
print("\n用户: 那我刚才问的假，哪个更长来着？")
print(f"助手: {agent.chat('那我刚才问的假，哪个更长来着？')}\n")

print("=" * 60)
print("代码做编排（流程/循环/记忆/错误），LLM 做决策（选工具/组答案）。")
print("=" * 60)
