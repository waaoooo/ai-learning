# alert_engine.py
from collections import defaultdict
from datetime import datetime

class AlertRule:
    def __init__(self, name: str, condition, severity: str, message: str, cooldown: int = 60):
        self.name = name
        self.condition = condition  # lambda metrics -> bool
        self.severity = severity    # critical/warning/info
        self.message = message
        self.cooldown = cooldown

class AlertEngine:
    def __init__(self):
        self.rules = []
        self.history = []
        self.last_alert = {}

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def evaluate(self, metrics: dict) -> list:
        triggered = []
        now = __import__("time").time()
        for rule in self.rules:
            if rule.name in self.last_alert:
                if now - self.last_alert[rule.name] < rule.cooldown:
                    continue
            try:
                if rule.condition(metrics):
                    alert = {
                        "rule": rule.name,
                        "severity": rule.severity,
                        "message": rule.message,
                        "time": datetime.now().strftime("%H:%M:%S"),
                    }
                    self.history.append(alert)
                    self.last_alert[rule.name] = now
                    triggered.append(alert)
            except Exception:
                pass
        return triggered

    def notify(self, alerts: list):
        icons = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
        for a in alerts:
            print(f"  {icons.get(a['severity'], '⚪')} [{a['severity'].upper()}] {a['rule']}: {a['message']}")

if __name__ == "__main__":
    engine = AlertEngine()
    engine.add_rule(AlertRule("high_error_rate", lambda m: m.get("error_rate", 0) > 0.05, "critical", "错误率超 5%", 60))
    engine.add_rule(AlertRule("high_latency", lambda m: m.get("p95_latency", 0) > 10, "critical", "P95 超 10s", 60))
    engine.add_rule(AlertRule("low_cache_hit", lambda m: m.get("cache_hit_rate", 1) < 0.3, "info", "缓存命中率低于 30%", 300))

    metrics = {"error_rate": 0.08, "p95_latency": 12.5, "cache_hit_rate": 0.25}
    triggered = engine.evaluate(metrics)
    print(f"触发 {len(triggered)} 个告警:")
    engine.notify(triggered)
