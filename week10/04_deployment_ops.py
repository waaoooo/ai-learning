"""
W10 第 4 课：部署与运维

演示：
  ① 健康检查端点（存活/就绪）
  ② 配置管理（环境变量 + Feature Flag）
  ③ 结构化日志 + 请求追踪
  ④ 性能监控（Prometheus 指标）
  ⑤ 告警规则引擎
  ⑥ 自动扩缩容模拟
"""
import os
import json
import time
import threading
import random
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 1.0


# ══════════════════════════════════════════════════════════
# ① 健康检查端点
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 健康检查端点")
print("=" * 60)


class HealthChecker:
    """健康检查：存活 + 就绪"""

    def __init__(self):
        self.checks = {
            "database": True,
            "redis": True,
            "llm_api": True,
            "disk_space": True,
            "memory": True,
        }
        self.start_time = time.time()

    def liveness(self) -> dict:
        """存活检查：进程是否运行"""
        return {
            "status": "alive",
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "timestamp": datetime.now().isoformat(),
        }

    def readiness(self) -> dict:
        """就绪检查：是否可以接收流量"""
        all_ok = all(self.checks.values())
        return {
            "status": "ready" if all_ok else "not_ready",
            "checks": self.checks,
            "timestamp": datetime.now().isoformat(),
        }

    def set_check(self, name: str, status: bool):
        self.checks[name] = status

    def simulate_failure(self, component: str):
        """模拟组件故障"""
        self.checks[component] = False
        print(f"  ⚠️ {component} 故障")


hc = HealthChecker()

print("  存活检查 (liveness):")
print(f"    {hc.liveness()}")

print("\n  就绪检查 (readiness):")
print(f"    {hc.readiness()}")

# 模拟故障
hc.simulate_failure("redis")
print("\n  Redis 故障后:")
print(f"    就绪: {hc.readiness()['status']}")
print(f"    检查: {hc.readiness()['checks']}")

# 恢复
hc.set_check("redis", True)
print("\n  Redis 恢复后:")
print(f"    就绪: {hc.readiness()['status']}")


# ══════════════════════════════════════════════════════════
# ② 配置管理
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 配置管理 (Feature Flags)")
print("=" * 60)


class ConfigManager:
    """配置管理：环境变量 + Feature Flags"""

    def __init__(self):
        self.config = {
            "model": os.getenv("MODEL", "glm-4-flash"),
            "max_tokens": int(os.getenv("MAX_TOKENS", "200")),
            "temperature": float(os.getenv("TEMPERATURE", "0.0")),
            "rate_limit": int(os.getenv("RATE_LIMIT", "60")),
        }
        self.feature_flags = {
            "new_rag_pipeline": True,
            "semantic_cache": True,
            "multi_model_routing": False,
            "advanced_audit": True,
        }
        self.flag_overrides = {}  # 运行时覆盖

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def is_enabled(self, flag: str, user_id: str = None) -> bool:
        """Feature Flag 检查（支持按用户灰度）"""
        if flag in self.flag_overrides:
            return self.flag_overrides[flag]
        base = self.feature_flags.get(flag, False)
        # 灰度：10% 用户开启
        if base and user_id:
            import hashlib
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
            return hash_val < 10
        return base

    def override(self, flag: str, value: bool):
        """运行时覆盖 Feature Flag"""
        self.flag_overrides[flag] = value
        print(f"  🔧 Feature Flag '{flag}' → {value}")


cm = ConfigManager()

print("  配置:")
for k, v in cm.config.items():
    print(f"    {k}: {v}")

print("\n  Feature Flags:")
for flag in cm.feature_flags:
    print(f"    {flag}: {cm.is_enabled(flag)}")

# 灰度测试
print("\n  灰度测试 (multi_model_routing, 10% 用户):")
enabled_users = []
for i in range(20):
    uid = f"user-{i}"
    if cm.is_enabled("multi_model_routing", uid):
        enabled_users.append(uid)
print(f"    20 个用户中 {len(enabled_users)} 个开启: {enabled_users}")

# 运行时覆盖
cm.override("multi_model_routing", True)
print(f"    覆盖后: {cm.is_enabled('multi_model_routing')}")


# ══════════════════════════════════════════════════════════
# ③ 结构化日志 + 请求追踪
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 结构化日志 + 请求追踪")
print("=" * 60)


class StructuredLogger:
    """结构化日志：JSON 格式 + request_id 追踪"""

    def __init__(self):
        self.logs = []

    def log(self, level: str, message: str, request_id: str = None, **extra):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "request_id": request_id,
            **extra,
        }
        self.logs.append(entry)
        # 模拟输出
        print(f"    [{level}] {message}" + (f" (req={request_id})" if request_id else ""))

    def query_by_request(self, request_id: str) -> list:
        """按 request_id 查询日志"""
        return [e for e in self.logs if e.get("request_id") == request_id]

    def query_by_level(self, level: str) -> list:
        return [e for e in self.logs if e.get("level") == level]


logger = StructuredLogger()

# 模拟请求链路
req_id = "req-abc123"
logger.log("INFO", "请求接收", request_id=req_id, method="POST", path="/chat")
logger.log("INFO", "意图分类", request_id=req_id, complexity="simple", latency_ms=45)
logger.log("INFO", "模型调用", request_id=req_id, model="glm-4-flash", tokens=120)
logger.log("INFO", "响应返回", request_id=req_id, status=200, total_latency_ms=230)

# 查询特定请求的日志
print(f"\n  请求 {req_id} 的完整链路:")
for entry in logger.query_by_request(req_id):
    print(f"    {entry['timestamp'][:19]} {entry['message']}")


# ══════════════════════════════════════════════════════════
# ④ 性能监控（Prometheus 指标）
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 性能监控 (Prometheus 指标)")
print("=" * 60)


class MetricsCollector:
    """Prometheus 风格指标收集器"""

    def __init__(self):
        self.counters = defaultdict(int)       # 计数器
        self.gauges = {}                      # 仪表盘
        self.histograms = defaultdict(list)    # 直方图

    def inc_counter(self, name: str, labels: dict = None, value: int = 1):
        key = f"{name}{labels or ''}"
        self.counters[key] += value

    def set_gauge(self, name: str, value: float, labels: dict = None):
        key = f"{name}{labels or ''}"
        self.gauges[key] = value

    def observe_histogram(self, name: str, value: float, labels: dict = None):
        key = f"{name}{labels or ''}"
        self.histograms[key].append(value)

    def get_percentile(self, name: str, p: float) -> float:
        values = sorted(self.histograms.get(name, []))
        if not values:
            return 0
        idx = int(len(values) * p / 100)
        return values[min(idx, len(values) - 1)]

    def export(self) -> str:
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

# 模拟请求指标
random.seed(42)
for i in range(50):
    latency = random.uniform(0.5, 5.0)
    tokens = random.randint(50, 300)
    status = "200" if random.random() > 0.05 else "500"

    metrics.inc_counter("http_requests_total", {"status": status})
    metrics.observe_histogram("http_request_duration_seconds", latency)
    metrics.observe_histogram("llm_tokens_per_request", tokens)
    metrics.inc_counter("llm_api_calls_total")

metrics.set_gauge("active_connections", 42)
metrics.set_gauge("cache_hit_rate", 0.65)

print("  指标统计:")
print(f"    总请求数: {sum(v for k, v in metrics.counters.items() if 'http_requests' in k)}")
print(f"    P50 延迟: {metrics.get_percentile('http_request_duration_seconds', 50):.2f}s")
print(f"    P95 延迟: {metrics.get_percentile('http_request_duration_seconds', 95):.2f}s")
print(f"    P99 延迟: {metrics.get_percentile('http_request_duration_seconds', 99):.2f}s")
print(f"    平均 Token: {sum(metrics.histograms['llm_tokens_per_request'])/len(metrics.histograms['llm_tokens_per_request']):.0f}")

print("\n  Prometheus 导出格式 (节选):")
export = metrics.export()
for line in export.split("\n")[:8]:
    print(f"    {line}")


# ══════════════════════════════════════════════════════════
# ⑤ 告警规则引擎
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑤ 告警规则引擎")
print("=" * 60)


@dataclass
class AlertRule:
    name: str
    condition: Callable
    severity: str  # critical/warning/info
    message: str


class AlertEngine:
    """告警引擎：规则评估 + 通知"""

    def __init__(self):
        self.rules = []
        self.alerts = []

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def evaluate(self, metrics: dict):
        """评估所有规则"""
        triggered = []
        for rule in self.rules:
            try:
                if rule.condition(metrics):
                    alert = {
                        "rule": rule.name,
                        "severity": rule.severity,
                        "message": rule.message,
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.alerts.append(alert)
                    triggered.append(alert)
            except Exception:
                pass
        return triggered

    def notify(self, alerts: list):
        """发送通知"""
        for alert in alerts:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(alert["severity"], "⚪")
            print(f"    {icon} [{alert['severity'].upper()}] {alert['rule']}: {alert['message']}")


from dataclasses import dataclass

engine = AlertEngine()

# 定义告警规则
engine.add_rule(AlertRule(
    name="high_error_rate",
    condition=lambda m: m.get("error_rate", 0) > 0.05,
    severity="critical",
    message="错误率超过 5%",
))
engine.add_rule(AlertRule(
    name="high_latency",
    condition=lambda m: m.get("p95_latency", 0) > 10,
    severity="critical",
    message="P95 延迟超过 10s",
))
engine.add_rule(AlertRule(
    name="high_cost",
    condition=lambda m: m.get("cost_per_request", 0) > 0.01,
    severity="warning",
    message="单请求成本超过 0.01 元",
))
engine.add_rule(AlertRule(
    name="low_cache_hit",
    condition=lambda m: m.get("cache_hit_rate", 1) < 0.3,
    severity="info",
    message="缓存命中率低于 30%",
))
engine.add_rule(AlertRule(
    name="high_token_usage",
    condition=lambda m: m.get("avg_tokens", 0) > 2000,
    severity="warning",
    message="平均 Token 超过 2000",
))

# 评估
current_metrics = {
    "error_rate": 0.08,       # 8% 错误率 → 触发
    "p95_latency": 12.5,      # 12.5s → 触发
    "cost_per_request": 0.015, # 0.015 元 → 触发
    "cache_hit_rate": 0.25,    # 25% → 触发
    "avg_tokens": 1500,        # 1500 → 不触发
}

print("  当前指标:")
for k, v in current_metrics.items():
    print(f"    {k}: {v}")

triggered = engine.evaluate(current_metrics)
print(f"\n  触发 {len(triggered)} 个告警:")
engine.notify(triggered)


# ══════════════════════════════════════════════════════════
# ⑥ 自动扩缩容模拟
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑥ 自动扩缩容模拟 (HPA)")
print("=" * 60)


class AutoScaler:
    """模拟 Kubernetes HPA"""

    def __init__(self, min_replicas=2, max_replicas=10, target_cpu=70):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_cpu = target_cpu
        self.current_replicas = min_replicas

    def scale(self, current_cpu: float) -> int:
        """根据 CPU 使用率决定扩缩容"""
        if current_cpu > self.target_cpu * 1.2:  # 超过目标 20%
            new_replicas = min(self.max_replicas, self.current_replicas + 2)
            action = "扩容"
        elif current_cpu < self.target_cpu * 0.5:  # 低于目标 50%
            new_replicas = max(self.min_replicas, self.current_replicas - 1)
            action = "缩容"
        else:
            new_replicas = self.current_replicas
            action = "保持"

        if new_replicas != self.current_replicas:
            print(f"    CPU={current_cpu}% → {action}: {self.current_replicas} → {new_replicas} 副本")
        else:
            print(f"    CPU={current_cpu}% → {action}: {self.current_replicas} 副本")

        self.current_replicas = new_replicas
        return new_replicas


scaler = AutoScaler(min_replicas=2, max_replicas=10, target_cpu=70)

# 模拟负载变化
load_scenarios = [
    ("正常负载", 65),
    ("高峰负载", 95),
    ("高峰持续", 90),
    ("负载下降", 40),
    ("低谷", 20),
    ("恢复", 60),
]

for name, cpu in load_scenarios:
    print(f"  {name}:")
    scaler.scale(cpu)
    time.sleep(0.3)


# ══════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("W10 第 4 课总结")
print("=" * 60)
print("""
  ① 健康检查：存活 (liveness) + 就绪 (readiness)
  ② 配置管理：环境变量 + Feature Flags + 灰度发布
  ③ 结构化日志：JSON 格式 + request_id 全链路追踪
  ④ 性能监控：Prometheus 指标 + 百分位延迟
  ⑤ 告警引擎：规则评估 + 多级通知
  ⑥ 自动扩缩容：基于 CPU 的 HPA 模拟

  部署不是终点，运维才是长期挑战
  监控 + 告警 + 自愈 = 生产级可靠性
""")
print("=" * 60)
print("\n  问你一个问题（引向 W10 第 5 课）：")
print("  一个企业级 Agent 系统，如何做性能优化？")
print("  （缓存、批处理、异步、模型优化……）")
print("=" * 60)
