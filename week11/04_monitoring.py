"""
W11 第 4 课：监控与告警

演示：
  ① 指标收集（Prometheus 风格）
  ② 日志聚合（结构化日志 + 搜索）
  ③ 链路追踪（Trace/Span）
  ④ 告警规则引擎（多级告警）
  ⑤ 健康检查与自愈
  ⑥ 监控仪表盘（Grafana 风格）
"""
import os
import json
import time
import random
import threading
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 0.5


# ══════════════════════════════════════════════════════════
# ① 指标收集 (Metrics Collection)
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 指标收集 (Prometheus 风格)")
print("=" * 60)

class MetricsCollector:
    """Prometheus 风格指标收集器"""

    def __init__(self):
        self.counters = defaultdict(int)       # 计数器 (只增不减)
        self.gauges = {}                      # 仪表盘 (可增可减)
        self.histograms = defaultdict(list)    # 直方图 (分布)
        self.summaries = defaultdict(list)    # 摘要

    def inc(self, name: str, labels: dict = None, value: int = 1):
        key = self._key(name, labels)
        self.counters[key] += value

    def set_gauge(self, name: str, value: float, labels: dict = None):
        key = self._key(name, labels)
        self.gauges[key] = value

    def observe(self, name: str, value: float, labels: dict = None):
        key = self._key(name, labels)
        self.histograms[key].append(value)

    def _key(self, name: str, labels: dict = None) -> str:
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name

    def percentile(self, name: str, p: float) -> float:
        values = sorted(self.histograms.get(name, []))
        if not values:
            return 0
        idx = int(len(values) * p / 100)
        return values[min(idx, len(values) - 1)]

    def export_prometheus(self) -> str:
        """导出 Prometheus 格式"""
        lines = []
        for key, val in self.counters.items():
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {val}")
        for key, val in self.gauges.items():
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {val}")
        return "\n".join(lines)


metrics = MetricsCollector()

# 模拟 100 个请求
random.seed(42)
for i in range(100):
    latency = random.uniform(0.5, 8.0)
    tokens = random.randint(50, 500)
    status = "200" if random.random() > 0.05 else "500"
    model = random.choice(["glm-4-flash", "glm-4-plus"])

    metrics.inc("http_requests_total", {"status": status, "method": "POST"})
    metrics.observe("http_request_duration_seconds", latency, {"endpoint": "/chat"})
    metrics.observe("llm_tokens_per_request", tokens, {"model": model})
    metrics.inc("llm_api_calls_total", {"model": model})

metrics.set_gauge("active_connections", 42)
metrics.set_gauge("cache_hit_rate", 0.65)
metrics.set_gauge("queue_depth", 3)

print("  指标统计:")
total_200 = metrics.counters.get('http_requests_total{method="POST",status="200"}', 0)
total_500 = metrics.counters.get('http_requests_total{method="POST",status="500"}', 0)
print(f"    总请求数: {total_200 + total_500:6d}  成功: {total_200:6d}  失败: {total_500:6d}")
p50_val = metrics.percentile('http_request_duration_seconds{endpoint="/chat"}', 50)
p95_val = metrics.percentile('http_request_duration_seconds{endpoint="/chat"}', 95)
p99_val = metrics.percentile('http_request_duration_seconds{endpoint="/chat"}', 99)
print(f"    P50 延迟: {p50_val:.2f}s")
print(f"    P95 延迟: {p95_val:.2f}s")
print(f"    P99 延迟: {p99_val:.2f}s")
flash_tokens = sum(metrics.histograms.get('llm_tokens_per_request{model="glm-4-flash"}', []))
flash_count = len(metrics.histograms.get('llm_tokens_per_request{model="glm-4-flash"}', []))
avg_token = flash_tokens / flash_count if flash_count else 0
print(f"    平均 Token: {avg_token:.0f}")

print("\n  Prometheus 导出 (节选):")
for line in metrics.export_prometheus().split("\n")[:6]:
    print(f"    {line}")


# ══════════════════════════════════════════════════════════
# ② 日志聚合 (Log Aggregation)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 日志聚合 (Structured Logging)")
print("=" * 60)

class LogAggregator:
    """日志聚合器：收集、存储、搜索"""

    def __init__(self, max_size: int = 10000):
        self.logs = deque(maxlen=max_size)
        self.index = defaultdict(lambda: defaultdict(list))  # {field: {value: [log_indices]}}

    def ingest(self, log_entry: dict):
        """摄入日志"""
        idx = len(self.logs)
        self.logs.append(log_entry)
        # 建立索引
        for field in ["level", "service", "request_id", "user_id"]:
            if field in log_entry:
                self.index[field][log_entry[field]].append(idx)

    def search(self, **filters) -> list[dict]:
        """搜索日志"""
        if not filters:
            return list(self.logs)

        # 找到匹配的索引
        matching_indices = None
        for field, value in filters.items():
            indices = set(self.index.get(field, {}).get(value, []))
            if matching_indices is None:
                matching_indices = indices
            else:
                matching_indices &= indices

        if matching_indices is None:
            return []
        return [self.logs[i] for i in sorted(matching_indices)]

    def count(self, **filters) -> int:
        return len(self.search(**filters))

    def aggregate(self, field: str) -> dict:
        """聚合统计"""
        counts = defaultdict(int)
        for log in self.logs:
            val = log.get(field, "unknown")
            counts[val] += 1
        return dict(counts)


logger = LogAggregator()

# 生成模拟日志
services = ["chat-service", "rag-service", "llm-proxy", "tool-service"]
levels = ["INFO", "INFO", "INFO", "WARN", "ERROR"]
for i in range(200):
    log = {
        "timestamp": datetime.now().isoformat(),
        "level": random.choice(levels),
        "service": random.choice(services),
        "request_id": f"req-{random.randint(1, 50)}",
        "user_id": f"user-{random.randint(1, 20)}",
        "message": f"Request processed in {random.randint(100, 5000)}ms",
        "latency_ms": random.randint(100, 5000),
    }
    logger.ingest(log)

print("  日志统计:")
print(f"    总日志数: {len(logger.logs)}")
print(f"    按级别: {logger.aggregate('level')}")
print(f"    按服务: {logger.aggregate('service')}")

# 搜索示例
error_logs = logger.search(level="ERROR")
print(f"\n  ERROR 日志数: {len(error_logs)}")

req_logs = logger.search(request_id="req-5")
print(f"  req-5 的日志数: {len(req_logs)}")

service_errors = logger.search(service="llm-proxy", level="ERROR")
print(f"  llm-proxy 的 ERROR 数: {len(service_errors)}")


# ══════════════════════════════════════════════════════════
# ③ 链路追踪 (Distributed Tracing)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 链路追踪 (Distributed Tracing)")
print("=" * 60)

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str = None
    name: str = ""
    start_time: float = 0
    end_time: float = 0
    status: str = "ok"
    attributes: dict = field(default_factory=dict)

    @property
    def duration_ms(self):
        return round((self.end_time - self.start_time) * 1000, 1)


class Tracer:
    """分布式追踪器"""

    def __init__(self):
        self.traces = defaultdict(list)  # {trace_id: [spans]}
        self.active_spans = {}

    def start_span(self, trace_id: str, name: str, parent_id: str = None) -> str:
        span_id = f"span-{len(self.traces[trace_id]) + 1}"
        span = Span(trace_id=trace_id, span_id=span_id, parent_span_id=parent_id,
                    name=name, start_time=time.time())
        self.traces[trace_id].append(span)
        self.active_spans[span_id] = span
        return span_id

    def end_span(self, span_id: str, status: str = "ok", **attrs):
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.end_time = time.time()
            span.status = status
            span.attributes.update(attrs)
            del self.active_spans[span_id]

    def get_trace(self, trace_id: str) -> list[Span]:
        return self.traces.get(trace_id, [])

    def print_trace(self, trace_id: str):
        spans = self.get_trace(trace_id)
        # 构建树形结构
        span_map = {s.span_id: s for s in spans}
        roots = [s for s in spans if s.parent_span_id is None]

        def print_span(span, indent=0):
            icon = "✅" if span.status == "ok" else "❌"
            print(f"  {'  ' * indent}{icon} {span.name} ({span.duration_ms}ms)")
            children = [s for s in spans if s.parent_span_id == span.span_id]
            for child in children:
                print_span(child, indent + 1)

        for root in roots:
            print_span(root)


tracer = Tracer()

# 模拟一个请求的追踪
trace_id = "trace-001"
root = tracer.start_span(trace_id, "orchestrator")
time.sleep(0.1)

s1 = tracer.start_span(trace_id, "agent.product", root)
time.sleep(0.05)
llm1 = tracer.start_span(trace_id, "llm.call", s1)
time.sleep(0.08)
tracer.end_span(llm1, model="glm-4-flash", tokens=178)
tracer.end_span(s1)

s2 = tracer.start_span(trace_id, "agent.engineer", root)
time.sleep(0.06)
llm2 = tracer.start_span(trace_id, "llm.call", s2)
time.sleep(0.07)
tracer.end_span(llm2, model="glm-4-flash", tokens=177)
tracer.end_span(s2)

s3 = tracer.start_span(trace_id, "orchestrator.summarize", root)
time.sleep(0.1)
llm3 = tracer.start_span(trace_id, "llm.call", s3)
time.sleep(0.09)
tracer.end_span(llm3, model="glm-4-flash", tokens=293)
tracer.end_span(s3)

tracer.end_span(root)

print("  调用树:")
tracer.print_trace(trace_id)


# ══════════════════════════════════════════════════════════
# ④ 告警规则引擎 (Alerting)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 告警规则引擎")
print("=" * 60)

@dataclass
class AlertRule:
    name: str
    condition: callable
    severity: str  # critical/warning/info
    message: str
    cooldown: int = 300  # 冷却时间（秒）


class AlertEngine:
    """告警引擎：规则评估 + 通知 + 冷却"""

    def __init__(self):
        self.rules = []
        self.alert_history = []
        self.last_alert = {}  # {rule_name: timestamp}

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def evaluate(self, metrics: dict) -> list[dict]:
        """评估所有规则"""
        triggered = []
        now = time.time()
        for rule in self.rules:
            # 冷却检查
            if rule.name in self.last_alert:
                if now - self.last_alert[rule.name] < rule.cooldown:
                    continue

            try:
                if rule.condition(metrics):
                    alert = {
                        "rule": rule.name,
                        "severity": rule.severity,
                        "message": rule.message,
                        "timestamp": datetime.now().isoformat(),
                        "metrics": {k: metrics.get(k) for k in ["error_rate", "p95_latency", "cost"] if k in metrics},
                    }
                    self.alert_history.append(alert)
                    self.last_alert[rule.name] = now
                    triggered.append(alert)
            except Exception:
                pass
        return triggered

    def notify(self, alerts: list[dict], channels: list[str] = None):
        """发送通知"""
        channels = channels or ["slack"]
        for alert in alerts:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(alert["severity"], "⚪")
            for channel in channels:
                print(f"    {icon} [{channel}] {alert['severity'].upper()}: {alert['rule']} — {alert['message']}")

    def report(self):
        print(f"\n  告警统计: 总计 {len(self.alert_history)} 次")
        by_severity = defaultdict(int)
        for a in self.alert_history:
            by_severity[a["severity"]] += 1
        for sev, count in by_severity.items():
            print(f"    {sev}: {count} 次")


engine = AlertEngine()

# 定义告警规则
engine.add_rule(AlertRule("high_error_rate", lambda m: m.get("error_rate", 0) > 0.05, "critical", "错误率超过 5%", 60))
engine.add_rule(AlertRule("high_latency", lambda m: m.get("p95_latency", 0) > 10, "critical", "P95 延迟超过 10s", 60))
engine.add_rule(AlertRule("high_cost", lambda m: m.get("cost_per_request", 0) > 0.01, "warning", "单请求成本超过 0.01 元", 300))
engine.add_rule(AlertRule("low_cache_hit", lambda m: m.get("cache_hit_rate", 1) < 0.3, "info", "缓存命中率低于 30%", 600))
engine.add_rule(AlertRule("high_token_usage", lambda m: m.get("avg_tokens", 0) > 2000, "warning", "平均 Token 超过 2000", 300))

# 评估
current_metrics = {
    "error_rate": 0.08,
    "p95_latency": 12.5,
    "cost_per_request": 0.015,
    "cache_hit_rate": 0.25,
    "avg_tokens": 1500,
}

triggered = engine.evaluate(current_metrics)
print(f"  触发 {len(triggered)} 个告警:")
engine.notify(triggered, channels=["slack", "pagerduty"])
engine.report()


# ══════════════════════════════════════════════════════════
# ⑤ 健康检查与自愈
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑤ 健康检查与自愈")
print("=" * 60)

class HealthMonitor:
    """健康监控：存活 + 就绪 + 自愈"""

    def __init__(self):
        self.checks = {
            "database": {"status": True, "last_check": time.time()},
            "redis": {"status": True, "last_check": time.time()},
            "llm_api": {"status": True, "last_check": time.time()},
            "disk_space": {"status": True, "last_check": time.time()},
        }
        self.restart_count = defaultdict(int)

    def check_all(self) -> dict:
        """检查所有组件"""
        result = {"status": "healthy", "checks": {}}
        for name, info in self.checks.items():
            # 模拟：随机故障
            if random.random() < 0.1:  # 10% 概率故障
                info["status"] = False
            info["last_check"] = time.time()
            result["checks"][name] = info["status"]
            if not info["status"]:
                result["status"] = "degraded"
        return result

    def self_heal(self, check_result: dict):
        """自愈：重启故障组件"""
        for name, status in check_result["checks"].items():
            if not status:
                print(f"    🔧 自愈: 重启 {name}...")
                self.checks[name]["status"] = True
                self.restart_count[name] += 1
                print(f"    ✅ {name} 已恢复 (重启次数: {self.restart_count[name]})")

    def liveness(self) -> dict:
        """存活检查"""
        return {"status": "alive", "uptime": round(time.time() - self.start_time if hasattr(self, 'start_time') else 0, 1)}

    def readiness(self) -> dict:
        """就绪检查"""
        all_ok = all(info["status"] for info in self.checks.values())
        return {"status": "ready" if all_ok else "not_ready",
                "checks": {k: v["status"] for k, v in self.checks.items()}}


monitor = HealthMonitor()
monitor.start_time = time.time()

# 模拟 3 轮健康检查
for round_num in range(3):
    print(f"\n  第 {round_num + 1} 轮健康检查:")
    result = monitor.check_all()
    print(f"    状态: {result['status']}")
    print(f"    检查: {result['checks']}")
    if result["status"] != "healthy":
        monitor.self_heal(result)

print(f"\n  最终就绪状态: {monitor.readiness()}")


# ══════════════════════════════════════════════════════════
# ⑥ 监控仪表盘 (Dashboard)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑥ 监控仪表盘 (Grafana 风格)")
print("=" * 60)

class Dashboard:
    """监控仪表盘：多面板展示"""

    def __init__(self, metrics: MetricsCollector, logger: LogAggregator, tracer: Tracer):
        self.metrics = metrics
        self.logger = logger
        self.tracer = tracer

    def render(self):
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │  Agent 系统监控仪表盘                            │")
        print("  ├─────────────────────────────────────────────────┤")

        # Panel 1: 请求概览
        total_reqs = (self.metrics.counters.get('http_requests_total{method="POST",status="200"}', 0) +
                      self.metrics.counters.get('http_requests_total{method="POST",status="500"}', 0))
        print(f"  │  📊 请求概览                                    │")
        succ = self.metrics.counters.get('http_requests_total{method="POST",status="200"}', 0)
        fail = self.metrics.counters.get('http_requests_total{method="POST",status="500"}', 0)
        print(f"  │     总请求: {total_reqs:6d}  成功: {succ:6d}  失败: {fail:6d}  │")

        # Panel 2: 延迟分布
        p50 = self.metrics.percentile('http_request_duration_seconds{endpoint="/chat"}', 50)
        p95 = self.metrics.percentile('http_request_duration_seconds{endpoint="/chat"}', 95)
        p99 = self.metrics.percentile('http_request_duration_seconds{endpoint="/chat"}', 99)
        print(f"  │  ⏱️ 延迟分布                                    │")
        print(f"  │     P50: {p50:5.2f}s  P95: {p95:5.2f}s  P99: {p99:5.2f}s              │")

        # Panel 3: Token 用量
        flash_tokens = sum(self.metrics.histograms.get('llm_tokens_per_request{model="glm-4-flash"}', []))
        plus_tokens = sum(self.metrics.histograms.get('llm_tokens_per_request{model="glm-4-plus"}', []))
        print(f"  │  🔤 Token 用量                                   │")
        print(f"  │     Flash: {flash_tokens:6d}  Plus: {plus_tokens:6d}  总计: {flash_tokens+plus_tokens:6d}   │")

        # Panel 4: 日志统计
        log_counts = self.logger.aggregate("level")
        print(f"  │  📝 日志统计                                    │")
        print(f"  │     INFO: {log_counts.get('INFO', 0):4d}  WARN: {log_counts.get('WARN', 0):4d}  ERROR: {log_counts.get('ERROR', 0):4d}              │")

        # Panel 5: 告警状态
        print(f"  │  🚨 告警状态                                    │")
        print(f"  │     活跃告警: {len(engine.alert_history):2d}  未解决: {len(engine.alert_history):2d}                      │")

        print("  └─────────────────────────────────────────────────┘")


dashboard = Dashboard(metrics, logger, tracer)
dashboard.render()

# ══════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("W11 第 4 课总结")
print("=" * 60)
print("""
  ① 指标收集：Counter/Gauge/Histogram 三种类型，Prometheus 格式
  ② 日志聚合：结构化日志 + 索引搜索 + 聚合统计
  ③ 链路追踪：Trace/Span 树形调用链，parent_span_id 嵌套
  ④ 告警引擎：规则评估 + 冷却 + 多渠道通知
  ⑤ 健康检查：存活/就绪 + 自愈（自动重启故障组件）
  ⑥ 监控仪表盘：多面板展示（请求/延迟/Token/日志/告警）

  监控是系统的"眼睛" —— 没有监控，你不知道系统是否健康
  告警是系统的"耳朵" —— 有问题时自动通知
  自愈是系统的"免疫系统" —— 自动恢复，减少人工干预
""")
print("=" * 60)
print("\n  问你一个问题（引向 W11 第 5 课）：")
print("  文档与 API 规范应该包含哪些内容？")
print("  （OpenAPI、架构文档、运维手册……）")
print("=" * 60)
