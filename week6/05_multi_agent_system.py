"""
W6 第 5 课：综合 —— 多 Agent 系统架构（五层）

① 交互层    MultiAgentSystem.chat()
② 调度层    Orchestrator（分解、分发、裁决、升级）
③ Agent 层  Agent（角色分工，只做一件事）
④ 通信层    MessageBus（结构化消息 + 日志）
⑤ 资源层    LLM（模型调用）
"""
import os
import json
import time
import uuid
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))


# ══════════════════════════════════════════════════════════
# ⑤ 资源层：LLM 调用
# ══════════════════════════════════════════════════════════
class LLM:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                             base_url="https://open.bigmodel.cn/api/paas/v4/")

    def call(self, messages, system=None) -> str:
        last_err: Exception = RuntimeError("call failed")
        for attempt in range(4):
            try:
                msgs = ([{"role": "system", "content": system}] if system else []) + messages
                resp = self.client.chat.completions.create(
                    model="glm-4.7-flash", messages=msgs, temperature=0, max_tokens=300,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise last_err


# ══════════════════════════════════════════════════════════
# ④ 通信层：结构化消息 + 消息总线
# ══════════════════════════════════════════════════════════
class Message:
    def __init__(self, from_agent, to_agent, msg_type, instruction="", context="", input_text=""):
        self.msg_id = uuid.uuid4().hex[:8]
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.type = msg_type              # TASK / RESULT / REVIEW / ESCALATE
        self.instruction = instruction
        self.context = context
        self.input = input_text
        self.output: str = ""
        self.status = "created"

    def to_prompt(self) -> str:
        parts = [f"【角色】{self.instruction}"]
        if self.context:
            parts.append(f"【背景资料】{self.context}")
        if self.input:
            parts.append(f"【上游产出】{self.input}")
        parts.append("请只输出你负责的部分，不要越界。")
        return "\n\n".join(parts)


class MessageBus:
    """消息总线：记录所有消息，可追溯"""
    def __init__(self):
        self.log: list = []

    def send(self, msg: Message) -> None:
        self.log.append(msg)
        print(f"      📨 [{msg.msg_id}] {msg.from_agent} → {msg.to_agent} ({msg.type})")


# ══════════════════════════════════════════════════════════
# ③ Agent 层：角色分工
# ══════════════════════════════════════════════════════════
class Agent:
    def __init__(self, name: str, role_prompt: str, llm: LLM, bus: MessageBus):
        self.name = name
        self.role_prompt = role_prompt
        self.llm = llm
        self.bus = bus

    def handle(self, msg: Message) -> Message:
        self.bus.send(msg)
        msg.output = self.llm.call([{"role": "user", "content": msg.to_prompt()}], system=self.role_prompt)
        msg.status = "done"
        return msg


# ══════════════════════════════════════════════════════════
# ② 调度层：总调度（大脑）
# ══════════════════════════════════════════════════════════
class Orchestrator:
    def __init__(self, llm: LLM, bus: MessageBus):
        self.llm = llm
        self.bus = bus
        # 可用 Agent 注册表（加角色只需加一项）
        self.agents = {
            "product": Agent("product_agent", "你是资深产品经理。只从产品角度分析：用户价值、场景、优先级、风险。不要谈技术实现。", llm, bus),
            "engineer": Agent("engineer_agent", "你是资深工程师。只从工程角度分析：技术方案、工作量、风险、依赖。不要谈产品价值。", llm, bus),
            "tester": Agent("tester_agent", "你是资深测试工程师。只从测试角度分析：测试点、风险场景、验收标准。", llm, bus),
        }

    def decompose(self, task: str) -> list:
        """分解任务：决定要派给哪些 Agent"""
        prompt = ("任务：" + task + "\n\n可用的分析角色：产品、工程、测试。"
                  "判断这个任务需要哪几个角色参与，输出 JSON：{\"roles\": [\"product\", \"engineer\"]}")
        r = self.llm.call(
            [{"role": "user", "content": prompt}],
            system="你是任务分解器，只输出 JSON。",
        )
        try:
            roles = json.loads(r).get("roles", ["product", "engineer"])
        except json.JSONDecodeError:
            roles = ["product", "engineer"]
        return [x for x in roles if x in self.agents]

    def dispatch(self, task: str, roles: list) -> dict:
        """分发给各 Agent（各自独立，无依赖）"""
        results = {}
        for role in roles:
            m = Message("orchestrator", self.agents[role].name, "TASK",
                        instruction=self.agents[role].role_prompt, context=task)
            results[role] = self.agents[role].handle(m).output
        return results

    def detect_conflict(self, results: dict) -> bool:
        """检测是否存在分歧"""
        if len(results) < 2:
            return False
        r = self.llm.call(
            [{"role": "user", "content": f"各方结论：{json.dumps(results, ensure_ascii=False)}\n\n"
                                          f"这些结论是否存在实质性分歧？只回答 YES 或 NO。"}],
            system="你是冲突检测器。",
        )
        return "YES" in r.upper()

    def arbitrate(self, task: str, results: dict) -> str:
        """仲裁：中立第三方综合评估"""
        m = Message("orchestrator", "arbiter_agent", "REVIEW",
                    instruction="你是中立的项目顾问，综合各方观点给出折中方案，不偏向任何一方。",
                    context=task, input_text=json.dumps(results, ensure_ascii=False))
        arbiter = Agent("arbiter_agent", m.instruction, self.llm, self.bus)
        return arbiter.handle(m).output

    def escalate(self, task: str, results: dict) -> str:
        """升级人工：整理成给决策者的报告"""
        m = Message("orchestrator", "escalator", "ESCALATE",
                    instruction="你是项目调度者，把无法调和的分歧整理成决策报告，列出成本收益和需要决策者拍板的问题。不要替决策者做决定。",
                    context=task, input_text=json.dumps(results, ensure_ascii=False))
        escalator = Agent("escalator", m.instruction, self.llm, self.bus)
        return escalator.handle(m).output

    def run(self, task: str) -> str:
        print("  【调度】任务分解")
        roles = self.decompose(task)
        print(f"      需要角色: {roles}")

        print("\n  【调度】分发给各 Agent（各自独立）")
        results = self.dispatch(task, roles)
        for k, v in results.items():
            print(f"      {k}: {v[:60].replace(chr(10),' ')}...")

        print("\n  【调度】冲突检测")
        if self.detect_conflict(results):
            print("      ⚠️ 检测到分歧 → 启动仲裁")
            return self.arbitrate(task, results)
        print("      ✅ 无实质分歧 → 直接汇总")
        return self._summarize(task, results)

    def _summarize(self, task: str, results: dict) -> str:
        return self.llm.call(
            [{"role": "user", "content": f"任务：{task}\n\n各方结论：{json.dumps(results, ensure_ascii=False)}\n\n汇总成一份简报。"}],
            system="你是项目总调度，负责汇总。",
        )


# ══════════════════════════════════════════════════════════
# ① 交互层：对外接口
# ══════════════════════════════════════════════════════════
class MultiAgentSystem:
    def __init__(self):
        self.llm = LLM()
        self.bus = MessageBus()
        self.orchestrator = Orchestrator(self.llm, self.bus)
        self.sessions = {}

    def chat(self, session_id: str, task: str) -> str:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        answer = self.orchestrator.run(task)
        self.sessions[session_id].append({"task": task, "answer": answer})
        return answer


# ══════════════════════════════════════════════════════════
# 运行
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("多 Agent 系统（五层架构）")
    print("=" * 60)
    system = MultiAgentSystem()

    TASK = "评估需求：给问答系统加语音输入功能（团队 2 人，本季度剩余工期 1 个月）"
    print(f"\n用户: {TASK}\n")
    ans = system.chat("demo", TASK)

    print("\n" + "=" * 60)
    print("最终输出")
    print("=" * 60)
    print(ans[:700])

    print("\n" + "=" * 60)
    print(f"消息总线日志：共 {len(system.bus.log)} 条消息")
    print("=" * 60)
    for m in system.bus.log:
        print(f"  [{m.msg_id}] {m.from_agent} → {m.to_agent} ({m.type}) {m.status}")
    print("\n✅ 全链路可追溯：每条消息都有 msg_id、收发方、类型、状态。")
