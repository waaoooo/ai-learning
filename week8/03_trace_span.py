"""
W8 第 3 课：链路追踪 — Trace 与 Span

演示：
  ① Trace/Span 数据模型（trace_id, span_id, parent_span_id）
  ② 多 Agent 链路的 Span 树（orchestrator → agent1, agent2, agent3）
  ③ 从 Span 列表还原调用树
  ④ 可视化 Span 树（缩进打印）
"""
import os
import json
import time
import uuid
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 3.0


# ══════════════════════════════════════════════════════════
# ① Trace / Span 数据模型
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① Trace / Span 数据模型")
print("=" * 60)

print("""
  Trace = 一次完整请求的端到端追踪（一个 trace_id）
  Span  = Trace 内的一个工作单元（一个 span_id，可嵌套）

  字段：
    trace_id        : 唯一标识整条链路
    span_id         : 唯一标识这个工作单元
    parent_span_id  : 父 Span 的 ID（root span 的 parent 为 None）
    name            : 工作单元名称（如 "agent.product", "llm.call"）
    start_time      : 开始时间
    end_time        : 结束时间
    duration_ms     : 耗时
    status          : ok / error
    attributes      : 额外属性（model, tokens, input, output...）

  关系：
    ┌─────────────────────────────────────────────┐
    │ Trace (trace_id=abc123)                      │
    │  ┌───────────────────────────────────────┐  │
    │  │ Span A: orchestrator (root, no parent) │  │
    │  │  ┌─────────────────────────────────┐  │  │
    │  │  │ Span B: agent.product (child of A) │  │  │
    │  │  │  ┌───────────────────────────┐  │  │  │
    │  │  │  │ Span C: llm.call (child of B) │  │  │  │
    │  │  │  └───────────────────────────┘  │  │  │
    │  │  └─────────────────────────────────┘  │  │
    │  │  ┌─────────────────────────────────┐  │  │
    │  │  │ Span D: agent.engineer (child of A)│  │  │
    │  │  └─────────────────────────────────┘  │  │
    │  └───────────────────────────────────────┘  │
    └─────────────────────────────────────────────┘
""")


# ══════════════════════════════════════════════════════════
# ② Span 类与 Tracer
# ══════════════════════════════════════════════════════════
class Span:
    def __init__(self, trace_id, name, parent_span_id=None):
        self.trace_id = trace_id
        self.span_id = uuid.uuid4().hex[:8]
        self.parent_span_id = parent_span_id
        self.name = name
        self.start_time = time.time()
        self.end_time = None
        self.status = "running"
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def end(self, status="ok"):
        self.end_time = time.time()
        self.status = status

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_ms": round(self.start_time * 1000),
            "end_ms": round(self.end_time * 1000) if self.end_time else None,
            "duration_ms": round((self.end_time - self.start_time) * 1000) if self.end_time else None,
            "status": self.status,
            "attributes": self.attributes,
        }


class Tracer:
    """链路追踪器：管理 Span 的创建和嵌套"""

    def __init__(self, trace_id=None):
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.spans = []
        self._stack = []   # 当前 Span 栈（用于嵌套）

    def start_span(self, name, **attrs):
        parent = self._stack[-1].span_id if self._stack else None
        span = Span(self.trace_id, name, parent)
        for k, v in attrs.items():
            span.set_attribute(k, v)
        self.spans.append(span)
        self._stack.append(span)
        return span

    def end_span(self, span, status="ok"):
        span.end(status)
        if self._stack and self._stack[-1] is span:
            self._stack.pop()

    def dump(self, filepath=None):
        data = [s.to_dict() for s in self.spans]
        if filepath:
            with open(filepath, "a") as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return data


def call_llm(prompt: str, tracer: Tracer, parent_span: Span) -> str:
    """带追踪的 LLM 调用"""
    span = tracer.start_span("llm.call", model=MODEL, prompt=prompt[:50])
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=150,
            extra_body={"thinking": {"type": "disabled"}},
        )
        result = (resp.choices[0].message.content or "").strip()
        span.set_attribute("output", result[:50])
        tracer.end_span(span, "ok")
        return result
    except Exception as e:
        span.set_attribute("error", str(e)[:50])
        tracer.end_span(span, "error")
        raise


# ══════════════════════════════════════════════════════════
# ③ 多 Agent 链路追踪演示
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("③ 多 Agent 链路追踪演示")
print("=" * 60)

tracer = Tracer()
Q = "评估需求：给问答系统加语音输入功能"


def agent_product(question, tracer):
    span = tracer.start_span("agent.product")
    result = call_llm(f"你是产品经理。任务：{question}\n只从产品角度分析。", tracer, span)
    span.set_attribute("output", result[:50])
    tracer.end_span(span)
    return result


def agent_engineer(question, tracer):
    span = tracer.start_span("agent.engineer")
    result = call_llm(f"你是工程师。任务：{question}\n只从工程角度分析。", tracer, span)
    span.set_attribute("output", result[:50])
    tracer.end_span(span)
    return result


def agent_tester(question, tracer):
    span = tracer.start_span("agent.tester")
    result = call_llm(f"你是测试工程师。任务：{question}\n只从测试角度分析。", tracer, span)
    span.set_attribute("output", result[:50])
    tracer.end_span(span)
    return result


def orchestrator(question, tracer):
    """总调度：创建 root span，分发给各 Agent"""
    root = tracer.start_span("orchestrator", question=question[:50])

    # 分发给 3 个 Agent（各自创建子 Span）
    r1 = agent_product(question, tracer)
    time.sleep(GAP)
    r2 = agent_engineer(question, tracer)
    time.sleep(GAP)
    r3 = agent_tester(question, tracer)
    time.sleep(GAP)

    # 汇总
    summary_span = tracer.start_span("orchestrator.summarize")
    summary = call_llm(f"汇总以下三方分析：\n产品：{r1[:80]}\n工程：{r2[:80]}\n测试：{r3[:80]}\n输出一段总结。", tracer, summary_span)
    tracer.end_span(summary_span)

    tracer.end_span(root, "ok")
    return summary


print(f"\n  Trace ID: {tracer.trace_id}")
print(f"  问题: {Q}\n")

result = orchestrator(Q, tracer)
print(f"\n  最终汇总: {result[:100]}...")

# 保存 Span 数据
spans_data = tracer.dump("/tmp/trace_spans.jsonl")
print(f"\n  共 {len(spans_data)} 个 Span，已写入 /tmp/trace_spans.jsonl")


# ══════════════════════════════════════════════════════════
# ④ 从 Span 列表还原调用树
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 从 Span 列表还原调用树")
print("=" * 60)


def build_tree(spans):
    """从扁平的 Span 列表构建树结构"""
    by_id = {s["span_id"]: s for s in spans}
    children = {}
    roots = []
    for s in spans:
        pid = s["parent_span_id"]
        if pid is None:
            roots.append(s)
        else:
            children.setdefault(pid, []).append(s)
    return roots, children


def print_tree(roots, children, indent=0):
    """递归打印 Span 树"""
    for span in roots:
        prefix = "  " * indent
        status_icon = "✅" if span["status"] == "ok" else "❌"
        dur = span["duration_ms"] or 0
        print(f"{prefix}{status_icon} {span['name']} ({dur}ms) [span={span['span_id']}]")
        # 打印 attributes
        attrs = span.get("attributes", {})
        if attrs:
            for k, v in attrs.items():
                if k in ("prompt", "output"):
                    print(f"{prefix}   └─ {k}: {str(v)[:60]}")
        # 递归子节点
        kids = children.get(span["span_id"], [])
        print_tree(kids, children, indent + 1)


roots, children = build_tree(spans_data)
print_tree(roots, children)

# ══════════════════════════════════════════════════════════
# ⑤ 总结
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("关键设计要点：")
print("  1. Trace = 一次完整请求（一个 trace_id）")
print("  2. Span = 一个工作单元（span_id + parent_span_id）")
print("  3. parent_span_id 构建树结构 → 还原调用链")
print("  4. 每个 Span 记录：name, duration, status, attributes")
print("  5. attributes 存 input/output/model/tokens 等调试信息")
print("  6. 可视化：缩进打印 = 调用树；Grafana/Jaeger = 图形化")
print("=" * 60)
print("\n  查询示例（jq）：")
print("  # 找出 trace 的 root span")
print("  cat trace_spans.jsonl | jq 'select(.parent_span_id==null)'")
print("  # 找出所有 LLM 调用")
print("  cat trace_spans.jsonl | jq 'select(.name==\"llm.call\")'")
print("  # 找出耗时最长的 Span")
print("  cat trace_spans.jsonl | jq -s 'max_by(.duration_ms)'")
print("=" * 60)
