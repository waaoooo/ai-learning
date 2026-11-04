"""
W8 第 5 课：综合 —— 可观测性仪表盘

把 W8 的三大能力整合成一个统一的可观测性系统：
  ① 结构化日志（W8-T2）
  ② Trace/Span 链路追踪（W8-T3）
  ③ 性能追踪（W8-T4）

仪表盘视图：
  1. Token 用量面板
  2. 调用链路追踪面板
  3. 准确率面板
  4. 性能监控面板
  5. 错误/报警日志面板
  6. 资源使用面板
"""
import os
import json
import time
import uuid
import statistics
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 3.0


# ══════════════════════════════════════════════════════════
# 统一可观测性系统
# ══════════════════════════════════════════════════════════
class ObservabilitySystem:
    """统一的可观测性系统：日志 + 追踪 + 性能"""

    def __init__(self):
        self.logs = []          # 结构化日志
        self.spans = []         # Span 列表
        self.metrics = []       # 性能指标
        self.alerts = []        # 告警记录
        self._span_stack = []

    # ── 日志 ──
    def log(self, level: str, message: str, **fields):
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "level": level,
            "message": message,
            **fields,
        }
        self.logs.append(entry)
        icon = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌", "DEBUG": "🔍"}.get(level, "📝")
        print(f"  {icon} [{level}] {message}")

    # ── 追踪 ──
    def start_span(self, name: str, trace_id: str, parent_id=None, **attrs):
        span = {
            "trace_id": trace_id,
            "span_id": uuid.uuid4().hex[:8],
            "parent_span_id": parent_id,
            "name": name,
            "start_time": time.time(),
            "status": "running",
            "attributes": attrs,
        }
        self.spans.append(span)
        self._span_stack.append(span)
        return span

    def end_span(self, span, status="ok", **attrs):
        span["end_time"] = time.time()
        span["duration_ms"] = round((span["end_time"] - span["start_time"]) * 1000)
        span["status"] = status
        span["attributes"].update(attrs)
        if self._span_stack and self._span_stack[-1] is span:
            self._span_stack.pop()

    # ── 性能 ──
    def record_metric(self, name: str, value: float, unit: str = "", tags=None):
        self.metrics.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "name": name,
            "value": value,
            "unit": unit,
            "tags": tags or {},
        })

    # ── 告警 ──
    def check_alerts(self, thresholds: dict):
        for rule_name, (metric_name, threshold, comparator) in thresholds.items():
            values = [m["value"] for m in self.metrics if m["name"] == metric_name]
            if not values:
                continue
            latest = values[-1]
            triggered = (comparator == "gt" and latest > threshold) or \
                        (comparator == "lt" and latest < threshold)
            if triggered:
                alert = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "rule": rule_name,
                    "metric": metric_name,
                    "value": latest,
                    "threshold": threshold,
                    "severity": "critical" if latest > threshold * 2 else "warning",
                }
                self.alerts.append(alert)
                print(f"  🚨 ALERT: {rule_name} — {metric_name}={latest} (threshold={threshold})")

    # ── 仪表盘 ──
    def dashboard(self):
        print("\n" + "═" * 70)
        print("  📊 可观测性仪表盘")
        print("═" * 70)

        # 1. Token 用量面板
        print("\n  ┌─────────────────────────────────────────────────┐")
        print("  │  1. Token 用量面板                              │")
        print("  ├─────────────────────────────────────────────────┤")
        token_metrics = [m for m in self.metrics if "token" in m["name"]]
        total_tokens = sum(m["value"] for m in token_metrics)
        print(f"  │  总 Token 消耗: {total_tokens:>8}                          │")
        for m in token_metrics:
            print(f"  │    {m['name']:<20} {m['value']:>6} tokens                │")
        print("  └─────────────────────────────────────────────────┘")

        # 2. 调用链路追踪面板
        print("\n  ┌─────────────────────────────────────────────────┐")
        print("  │  2. 调用链路追踪面板                            │")
        print("  ├─────────────────────────────────────────────────┤")
        traces = {}
        for s in self.spans:
            traces.setdefault(s["trace_id"], []).append(s)
        for tid, spans in traces.items():
            roots = [s for s in spans if s["parent_span_id"] is None]
            for root in roots:
                print(f"  │  Trace: {tid}                              │")
                self._print_span_tree(root, spans, indent=2)
        print("  └─────────────────────────────────────────────────┘")

        # 3. 准确率面板
        print("\n  ┌─────────────────────────────────────────────────┐")
        print("  │  3. 准确率面板                                  │")
        print("  ├─────────────────────────────────────────────────┤")
        acc_metrics = [m for m in self.metrics if "accuracy" in m["name"]]
        for m in acc_metrics:
            bar = "█" * int(m["value"] * 20)
            print(f"  │  {m['name']:<20} {m['value']:>6.1%}  {bar:<20} │")
        print("  └─────────────────────────────────────────────────┘")

        # 4. 性能监控面板
        print("\n  ┌─────────────────────────────────────────────────┐")
        print("  │  4. 性能监控面板                                │")
        print("  ├─────────────────────────────────────────────────┤")
        lat_metrics = [m for m in self.metrics if "latency" in m["name"]]
        if lat_metrics:
            vals = [m["value"] for m in lat_metrics]
            print(f"  │  P50: {statistics.median(vals):>8.0f} ms                          │")
            print(f"  │  P95: {sorted(vals)[int(len(vals)*0.95)]:>8.0f} ms                          │")
            print(f"  │  Max: {max(vals):>8.0f} ms                          │")
            print(f"  │  Min: {min(vals):>8.0f} ms                          │")
        cost_metrics = [m for m in self.metrics if "cost" in m["name"]]
        if cost_metrics:
            print(f"  │  总成本: {sum(m['value'] for m in cost_metrics):>8.6f} 元                    │")
        print("  └─────────────────────────────────────────────────┘")

        # 5. 错误/报警日志面板
        print("\n  ┌─────────────────────────────────────────────────┐")
        print("  │  5. 错误/报警日志面板                            │")
        print("  ├─────────────────────────────────────────────────┤")
        errors = [l for l in self.logs if l["level"] in ("ERROR", "WARN")]
        if errors:
            for e in errors[-5:]:
                print(f"  │  [{e['level']}] {e['message'][:45]:<45} │")
        else:
            print("  │  ✅ 无错误                                      │")
        if self.alerts:
            for a in self.alerts[-3:]:
                print(f"  │  🚨 {a['rule']}: {a['metric']}={a['value']} (threshold={a['threshold']}) │")
        print("  └─────────────────────────────────────────────────┘")

        # 6. 资源使用面板
        print("\n  ┌─────────────────────────────────────────────────┐")
        print("  │  6. 资源使用面板                                │")
        print("  ├─────────────────────────────────────────────────┤")
        res_metrics = [m for m in self.metrics if "resource" in m["name"] or "cpu" in m["name"] or "mem" in m["name"]]
        if res_metrics:
            for m in res_metrics:
                print(f"  │  {m['name']:<20} {m['value']:>6.1f} {m['unit']:<6}              │")
        else:
            print("  │  (模拟数据) CPU: 45.2%  MEM: 1.2GB  GPU: 0%    │")
        print("  └─────────────────────────────────────────────────┘")

        print("\n" + "═" * 70)

    def _print_span_tree(self, root, all_spans, indent=2):
        prefix = "  " * indent
        status = "✅" if root["status"] == "ok" else "❌"
        dur = root.get("duration_ms", 0) or 0
        print(f"  │{prefix}{status} {root['name']} ({dur}ms)")
        children = [s for s in all_spans if s["parent_span_id"] == root["span_id"]]
        for c in children:
            self._print_span_tree(c, all_spans, indent + 1)


# ══════════════════════════════════════════════════════════
# 模拟一次完整的多 Agent 请求
# ══════════════════════════════════════════════════════════
print("=" * 70)
print("模拟一次完整的多 Agent 请求")
print("=" * 70)

obs = ObservabilitySystem()
trace_id = "trace-" + uuid.uuid4().hex[:8]
Q = "评估需求：给问答系统加语音输入功能"

# Root span
root = obs.start_span("orchestrator", trace_id, question=Q[:50])

# Agent 1: product
s1 = obs.start_span("agent.product", trace_id, parent_id=root["span_id"])
t0 = time.time()
r1 = client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": f"你是产品经理。任务：{Q}\n只从产品角度分析。"}],
    temperature=0, max_tokens=150, extra_body={"thinking": {"type": "disabled"}},
)
lat1 = (time.time() - t0) * 1000
out1 = (r1.choices[0].message.content or "").strip()
obs.end_span(s1, "ok", output=out1[:50], tokens=r1.usage.total_tokens if r1.usage else 0)
obs.log("INFO", "agent.product completed", req_id=trace_id, step="product", ms=round(lat1))
obs.record_metric("token.product", r1.usage.total_tokens if r1.usage else len(out1)//4, "tokens")
obs.record_metric("latency.product", lat1, "ms")
time.sleep(GAP)

# Agent 2: engineer
s2 = obs.start_span("agent.engineer", trace_id, parent_id=root["span_id"])
t0 = time.time()
r2 = client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": f"你是工程师。任务：{Q}\n只从工程角度分析。"}],
    temperature=0, max_tokens=150, extra_body={"thinking": {"type": "disabled"}},
)
lat2 = (time.time() - t0) * 1000
out2 = (r2.choices[0].message.content or "").strip()
obs.end_span(s2, "ok", output=out2[:50], tokens=r2.usage.total_tokens if r2.usage else 0)
obs.log("INFO", "agent.engineer completed", req_id=trace_id, step="engineer", ms=round(lat2))
obs.record_metric("token.engineer", r2.usage.total_tokens if r2.usage else len(out2)//4, "tokens")
obs.record_metric("latency.engineer", lat2, "ms")
time.sleep(GAP)

# Agent 3: tester
s3 = obs.start_span("agent.tester", trace_id, parent_id=root["span_id"])
t0 = time.time()
r3 = client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": f"你是测试工程师。任务：{Q}\n只从测试角度分析。"}],
    temperature=0, max_tokens=150, extra_body={"thinking": {"type": "disabled"}},
)
lat3 = (time.time() - t0) * 1000
out3 = (r3.choices[0].message.content or "").strip()
obs.end_span(s3, "ok", output=out3[:50], tokens=r3.usage.total_tokens if r3.usage else 0)
obs.log("INFO", "agent.tester completed", req_id=trace_id, step="tester", ms=round(lat3))
obs.record_metric("token.tester", r3.usage.total_tokens if r3.usage else len(out3)//4, "tokens")
obs.record_metric("latency.tester", lat3, "ms")
time.sleep(GAP)

# Summarize
s4 = obs.start_span("orchestrator.summarize", trace_id, parent_id=root["span_id"])
t0 = time.time()
r4 = client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": f"汇总：\n产品：{out1[:80]}\n工程：{out2[:80]}\n测试：{out3[:80]}\n输出一段总结。"}],
    temperature=0, max_tokens=200, extra_body={"thinking": {"type": "disabled"}},
)
lat4 = (time.time() - t0) * 1000
out4 = (r4.choices[0].message.content or "").strip()
obs.end_span(s4, "ok", output=out4[:50], tokens=r4.usage.total_tokens if r4.usage else 0)
obs.log("INFO", "orchestrator.summarize completed", req_id=trace_id, step="summarize", ms=round(lat4))
obs.record_metric("token.summarize", r4.usage.total_tokens if r4.usage else len(out4)//4, "tokens")
obs.record_metric("latency.summarize", lat4, "ms")

obs.end_span(root, "ok")
obs.log("INFO", "request completed", req_id=trace_id, total_ms=round(lat1+lat2+lat3+lat4))

# 模拟准确率数据
obs.record_metric("accuracy.single_agent", 0.72, "%")
obs.record_metric("accuracy.multi_agent", 0.86, "%")

# 模拟资源数据
obs.record_metric("resource.cpu", 45.2, "%")
obs.record_metric("resource.mem", 1.2, "GB")
obs.record_metric("resource.gpu", 0.0, "%")

# 模拟一个错误
obs.log("ERROR", "agent.tester timeout", req_id=trace_id, step="tester", ms=15000)
obs.record_metric("error.rate", 0.05, "%")

# 告警规则
obs.check_alerts({
    "high_latency": ("latency.product", 10000, "gt"),
    "high_cost": ("token.summarize", 500, "gt"),
    "high_error_rate": ("error.rate", 0.03, "gt"),
})

# ══════════════════════════════════════════════════════════
# 展示仪表盘
# ══════════════════════════════════════════════════════════
obs.dashboard()

# ══════════════════════════════════════════════════════════
# 保存数据
# ══════════════════════════════════════════════════════════
with open("/tmp/obs_dashboard.jsonl", "w") as f:
    for log in obs.logs:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
    for span in obs.spans:
        f.write(json.dumps(span, ensure_ascii=False, default=str) + "\n")
    for m in obs.metrics:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
    for a in obs.alerts:
        f.write(json.dumps(a, ensure_ascii=False) + "\n")

print(f"\n数据已保存到 /tmp/obs_dashboard.jsonl")
print(f"  日志: {len(obs.logs)} 条")
print(f"  Span: {len(obs.spans)} 个")
print(f"  指标: {len(obs.metrics)} 个")
print(f"  告警: {len(obs.alerts)} 条")
print("=" * 70)
print("\n关键设计要点：")
print("  1. 统一系统：日志 + 追踪 + 性能 + 告警 一体化")
print("  2. Trace/Span：链路追踪的骨架")
print("  3. 指标：Token/延迟/准确率/资源")
print("  4. 告警：基于阈值自动触发")
print("  5. 仪表盘：6 个面板，一屏概览")
print("  6. 生产环境：接入 Grafana/Prometheus/Jaeger")
print("=" * 70)
