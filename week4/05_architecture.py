"""
W4 第 5 课：综合 Agent 架构（五层）

① 交互层    —— UI/API、多轮对话
② 编排调度层 —— 规划 → 执行 → 反思
③ 资源层    —— 工具 / 知识库
④ 质量与安全层 —— 错误处理 / 重试
⑤ LLM 基础层 —— 模型调用（带限流重试）
"""
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))


# ══════════════════════════════════════════════════════════
# ⑤ LLM 基础层：只管"把消息发出去、拿回结果"
# ══════════════════════════════════════════════════════════
class LLMClient:
    def __init__(self, model="glm-4.7-flash", max_retries=4):
        self.client = OpenAI(
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )
        self.model = model
        self.max_retries = max_retries

    def call(self, messages, tools=None, json_mode=False):
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(model=self.model, messages=messages, temperature=0,
                              max_tokens=600,
                              extra_body={"thinking": {"type": "disabled"}})
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                return self.client.chat.completions.create(**kwargs)
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)


# ══════════════════════════════════════════════════════════
# ③ 资源层：工具 + 知识库（只负责"提供能力"）
# ══════════════════════════════════════════════════════════
DOCS_DB = {"年假": "正式员工年假15天。", "病假": "病假每年累计不超过30天。"}

def tool_get_weather(city: str) -> str:
    db = {"北京": "晴，22℃", "上海": "多云，25℃"}
    return f"{city}天气：{db[city]}" if city in db else f"ERROR: 未查到「{city}」"

def tool_search_docs(query: str) -> str:
    for k, v in DOCS_DB.items():
        if k in query:
            return v
    return f"ERROR: 未找到「{query}」相关制度"


class ResourceRegistry:
    """资源注册表：加工具只需加一项"""
    TOOLS = [
        {"name": "get_weather", "description": "查询城市天气",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
         "func": tool_get_weather},
        {"name": "search_docs", "description": "搜索公司制度",
         "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
         "func": tool_search_docs},
    ]

    def __init__(self):
        self.funcs = {t["name"]: t["func"] for t in self.TOOLS}

    def schemas(self):
        return [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in self.TOOLS]

    def execute(self, name, args):
        fn = self.funcs.get(name)
        if not fn:
            return f"ERROR: 未知工具 {name}"
        try:
            return fn(**args)
        except Exception as e:
            return f"ERROR: {e}"   # ④ 错误不崩溃，交给上层处理


# ══════════════════════════════════════════════════════════
# ② 编排调度层：规划 → 执行 → 反思
# ══════════════════════════════════════════════════════════
PLAN_SYSTEM = """你是任务规划器。把任务拆解成具体步骤。
你拥有工具：get_weather(city)、search_docs(query)。涉及数据查询的步骤必须指明用哪个工具。
只输出 JSON：{"steps": ["步骤1", ...]}"""

REFLECT_SYSTEM = """你是质检员。检查回答是否完整正确、有无遗漏或编造。
完全正确只输出 PASS；有问题输出 JSON：{"ok": false, "problems": [...], "fix": "..."}"""


class Orchestrator:
    def __init__(self, llm: LLMClient, resources: ResourceRegistry, max_steps=6):
        self.llm = llm
        self.resources = resources
        self.max_steps = max_steps

    def plan(self, task: str) -> list:
        r = self.llm.call([{"role": "system", "content": PLAN_SYSTEM},
                           {"role": "user", "content": task}], json_mode=True)
        return json.loads(r.choices[0].message.content)["steps"]

    def execute(self, step: str, context: str) -> str:
        messages = [
            {"role": "system", "content": "你是执行助手，只能用工具返回的信息回答，不要编造。"},
            {"role": "user", "content": f"已知信息：{context}\n\n完成这一步：{step}"},
        ]
        for _ in range(3):
            r = self.llm.call(messages, tools=self.resources.schemas())
            msg = r.choices[0].message
            messages.append(msg)
            if not msg.tool_calls:
                return msg.content
            for tc in msg.tool_calls:
                result = self.resources.execute(tc.function.name, json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return "（本步未完成）"

    def reflect(self, task: str, context: str, answer: str) -> dict:
        r = self.llm.call([{"role": "system", "content": REFLECT_SYSTEM},
                           {"role": "user", "content": f"任务：{task}\n已知信息：{context}\n回答：\n{answer}"}])
        out = r.choices[0].message.content.strip()
        if out.upper().startswith("PASS"):
            return {"ok": True}
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {"ok": True}   # 解析失败不阻塞

    def run(self, task: str, verbose=True) -> str:
        steps = self.plan(task)
        if verbose:
            print("  【规划】")
            for i, s in enumerate(steps, 1):
                print(f"    步骤{i}: {s}")

        context, results = "", []
        for s in steps:
            r = self.execute(s, context)
            results.append(r)
            context += f"\n- {s} → {r}"

        answer = self.llm.call(
            [{"role": "system", "content": "根据步骤结果给用户简洁完整的回答。"},
             {"role": "user", "content": f"任务：{task}\n\n结果：{context}"}]
        ).choices[0].message.content

        if verbose:
            print(f"\n  【初稿】{answer[:80]}...")

        critique = self.reflect(task, context, answer)   # ④ 反思
        if verbose:
            print(f"  【反思】{'PASS' if critique.get('ok') else critique}")

        if not critique.get("ok"):
            answer = self.llm.call(
                [{"role": "system", "content": "根据指出的问题修正回答。"},
                 {"role": "user", "content": f"任务：{task}\n原回答：{answer}\n问题：{critique}"}]
            ).choices[0].message.content
        return answer


# ══════════════════════════════════════════════════════════
# ① 交互层：对外接口 + 多轮对话
# ══════════════════════════════════════════════════════════
class AgentAPI:
    def __init__(self):
        self.llm = LLMClient()                        # ⑤
        self.resources = ResourceRegistry()           # ③
        self.orchestrator = Orchestrator(self.llm, self.resources)  # ②
        self.sessions = {}                            # ① 多轮对话

    def chat(self, session_id: str, message: str) -> str:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        history = self.sessions[session_id]
        history.append({"role": "user", "content": message})

        answer = self.orchestrator.run(message)
        history.append({"role": "assistant", "content": answer})
        return answer


# ══════════════════════════════════════════════════════════
# 运行
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("五层 Agent 架构演示")
print("=" * 60)
api = AgentAPI()

TASK = "查一下北京天气，再查公司年假天数，告诉我请3天假还剩几天。"
print(f"\n用户: {TASK}\n")
ans = api.chat("demo", TASK)
print(f"\n最终回答:\n{ans}")
print("\n" + "=" * 60)
print("交互层 → 编排层 → 资源层 → 质量层 → LLM 基础层")
print("=" * 60)
