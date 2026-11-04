# metrics_collector.py
from collections import defaultdict

class MetricsCollector:
    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = {}
        self.histograms = defaultdict(list)

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
            ls = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{ls}}}"
        return name

    def percentile(self, name: str, p: float) -> float:
        vals = sorted(self.histograms.get(name, []))
        if not vals:
            return 0
        idx = int(len(vals) * p / 100)
        return vals[min(idx, len(vals) - 1)]

if __name__ == "__main__":
    m = MetricsCollector()
    import random
    random.seed(42)
    for _ in range(100):
        latency = random.uniform(0.5, 8.0)
        status = "200" if random.random() > 0.05 else "500"
        m.inc("http_requests_total", {"status": status})
        m.observe("http_request_duration_seconds", latency)

    m.set_gauge("active_connections", 42)
    m.set_gauge("cache_hit_rate", 0.65)

    # 替换最后的 print 语句
    total_200 = m.counters.get('http_requests_total{status="200"}', 0)
    total_500 = m.counters.get('http_requests_total{status="500"}', 0)
    print(f"总请求: {total_200 + total_500}")
    print(f"P50: {m.percentile('http_request_duration_seconds', 50):.2f}s")
    print(f"P95: {m.percentile('http_request_duration_seconds', 95):.2f}s")
    print(f"活跃连接: {m.gauges['active_connections']}")
    print(f"缓存命中率: {m.gauges['cache_hit_rate']}")
