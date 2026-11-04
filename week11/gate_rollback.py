# gate_rollback.py
import random
import time

class GateChecker:
    """门禁检查"""
    GATES = {
        "lint_to_test": lambda r: r.get("lint") == "success",
        "test_to_build": lambda r: r.get("unit_test") == "success" and r.get("integration_test") == "success",
        "build_to_deploy": lambda r: r.get("image_scan") == "success",
        "staging_to_prod": lambda r: r.get("smoke_test") == "success",
    }

    def check(self, gate_name: str, results: dict) -> tuple:
        gate = self.GATES.get(gate_name)
        if not gate:
            return True, "无门禁"
        try:
            passed = gate(results)
            return passed, "通过" if passed else "阻断"
        except Exception as e:
            return False, f"异常: {e}"

class RollbackManager:
    """自动回滚"""
    def __init__(self):
        self.history = []
        self.current = "v1.0.0"

    def deploy(self, version: str) -> bool:
        print(f"  📦 部署 {version}...")
        time.sleep(0.3)
        success = random.random() > 0.3  # 70% 成功率
        if success:
            self.current = version
            self.history.append({"version": version, "status": "success"})
            print(f"  ✅ 成功")
        else:
            self.history.append({"version": version, "status": "failed"})
            print(f"  ❌ 失败，自动回滚")
            self.rollback()
        return success

    def rollback(self):
        for entry in reversed(self.history[:-1]):
            if entry["status"] == "success":
                self.current = entry["version"]
                print(f"  🔄 回滚到 {entry['version']}")
                return
        print("  ⚠️ 无成功版本可回滚")

if __name__ == "__main__":
    # 门禁演示
    gate = GateChecker()
    results = {"lint": "success", "unit_test": "success", "integration_test": "success",
               "image_scan": "success", "smoke_test": "success"}
    print("门禁检查:")
    for g in GateChecker.GATES:
        ok, msg = gate.check(g, results)
        print(f"  {'✅' if ok else '❌'} {g}: {msg}")

    # 回滚演示
    print("\n部署演示:")
    rb = RollbackManager()
    for v in ["v1.1.0", "v1.2.0", "v1.3.0"]:
        rb.deploy(v)
    print(f"\n最终版本: {rb.current}")
