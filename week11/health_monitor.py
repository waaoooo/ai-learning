# health_monitor.py
import random
import time

class HealthMonitor:
    def __init__(self):
        self.checks = {
            "database": True,
            "redis": True,
            "llm_api": True,
            "disk_space": True,
        }
        self.restart_count = {k: 0 for k in self.checks}

    def check_all(self) -> dict:
        result = {"status": "healthy", "checks": {}}
        for name in self.checks:
            if random.random() < 0.15:  # 15% 故障概率
                self.checks[name] = False
            result["checks"][name] = self.checks[name]
            if not self.checks[name]:
                result["status"] = "degraded"
        return result

    def self_heal(self, result: dict):
        for name, ok in result["checks"].items():
            if not ok:
                print(f"  🔧 重启 {name}...")
                self.checks[name] = True
                self.restart_count[name] += 1
                print(f"  ✅ {name} 已恢复 (重启 {self.restart_count[name]} 次)")

    def readiness(self) -> dict:
        all_ok = all(self.checks.values())
        return {"status": "ready" if all_ok else "not_ready",
                "checks": dict(self.checks)}

if __name__ == "__main__":
    monitor = HealthMonitor()
    for i in range(4):
        print(f"\n第 {i+1} 轮检查:")
        result = monitor.check_all()
        print(f"  状态: {result['status']}, 检查: {result['checks']}")
        if result["status"] != "healthy":
            monitor.self_heal(result)
    print(f"\n最终就绪: {monitor.readiness()}")

