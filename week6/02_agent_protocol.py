"""
W6 第 2 课：角色分工与通信协议

定义 Agent 间标准消息格式：
  {
    "msg_id":    唯一标识（可追溯）
    "from":      发送方
    "to":        接收方
    "type":      TASK / RESULT / QUESTION / REVIEW
    "instruction": 角色限定 + 提示词
    "context":   源数据
    "input":     上一个 agent 的产出
  }
"""
import os
import json
import time
import uuid
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/")


def call(messages, system=None):
    for attempt in range(4):
        try:
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            resp = client.chat.completions.create(
                model="glm-4.7-flash", messages=msgs, temperature=0, max_tokens=400,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


# ══════════════════════════════════════════════════════════
# ① 定义标准消息格式
# ══════════════════════════════════════════════════════════
class Message:
    """Agent 间通信的标准消息"""
    VALID_TYPES = ("TASK", "RESULT", "QUESTION", "REVIEW")

    def __init__(self, from_agent, to_agent, msg_type, instruction="", context="", input_text=""):
        assert msg_type in self.VALID_TYPES, f"未知消息类型: {msg_type}"
        self.msg_id = uuid.uuid4().hex[:8]
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.type = msg_type
        self.instruction = instruction      # 角色限定 + 提示词
        self.context = context              # 源数据
        self.input = input_text             # 上游产出
        self.output: str = ""                   # 下游填充
        self.status = "created"

    def to_prompt(self):
        """把消息转成给 LLM 的提示词"""
        parts = [f"【角色】{self.instruction}"]
        if self.context:
            parts.append(f"【背景资料】{self.context}")
        if self.input:
            parts.append(f"【上游产出】{self.input}")
        parts.append("请只输出你负责的部分，不要越界。")
        return "\n\n".join(parts)

    def dump(self):
        return {
            "msg_id": self.msg_id, "from": self.from_agent, "to": self.to_agent,
            "type": self.type, "instruction": self.instruction[:40] + "…",
            "context": self.context[:40] + "…", "input": (self.input or "")[:40] + "…",
            "output": (self.output or "")[:60] + "…", "status": self.status,
        }


# ══════════════════════════════════════════════════════════
# ② 定义 Agent 角色
# ══════════════════════════════════════════════════════════
class Agent:
    def __init__(self, name, role_prompt):
        self.name = name
        self.role_prompt = role_prompt

    def handle(self, msg: Message) -> Message:
        """接收消息 → 处理 → 把结果写回消息"""
        prompt = msg.to_prompt()
        result = call([{"role": "user", "content": prompt}], system=self.role_prompt)
        msg.output = result
        msg.status = "done"
        return msg


# ══════════════════════════════════════════════════════════
# ③ 演示：产品 Agent → 工程 Agent 的消息传递
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 定义标准消息格式")
print("=" * 60)

msg = Message(
    from_agent="orchestrator",
    to_agent="product_agent",
    msg_type="TASK",
    instruction="你是资深产品经理，只从产品角度分析：用户价值、场景、优先级、风险。不要谈技术实现。",
    context="需求：给问答系统加一个语音输入功能。",
)
print("\n  消息结构:")
print(json.dumps(msg.dump(), ensure_ascii=False, indent=2))

print("\n" + "=" * 60)
print("② 消息流转：产品 Agent → 工程 Agent")
print("=" * 60)

product_agent = Agent("product_agent",
    "你是一位资深产品经理，只从产品角度分析：用户价值、使用场景、优先级、风险。不要谈技术实现。")
engineer_agent = Agent("engineer_agent",
    "你是一位资深工程师，只从工程角度分析：技术方案、工作量、风险、依赖。不要谈产品价值。")

# 产品 Agent 处理 TASK
print("\n  【1】orchestrator 派发 TASK 给 product_agent")
m1 = product_agent.handle(msg)
print(f"      产品 Agent 产出: {m1.output[:80]}...")

# 把产品产出包装成新消息，传给工程 Agent
print("\n  【2】把产品产出作为 input，派发给 engineer_agent")
m2 = Message(
    from_agent="product_agent",
    to_agent="engineer_agent",
    msg_type="TASK",
    instruction="你是资深工程师，只从工程角度分析：技术方案、工作量、风险、依赖。不要谈产品价值。",
    context="需求：给问答系统加一个语音输入功能。",
    input_text=m1.output,      # ★ 上游产出作为下游输入
)
m2 = engineer_agent.handle(m2)
print(f"      工程 Agent 产出: {m2.output[:80]}...")

print("\n" + "=" * 60)
print("③ 消息日志（可追溯）")
print("=" * 60)
for m in (m1, m2):
    d = m.dump()
    print(f"  [{d['msg_id']}] {d['from']} → {d['to']} ({d['type']})  {d['status']}")

print("\n" + "=" * 60)
print("关键：消息是结构化的 —— 可追溯、可校验、可路由。")
print("      每个 Agent 只看到自己需要的信息，不会互相污染。")
print("=" * 60)
