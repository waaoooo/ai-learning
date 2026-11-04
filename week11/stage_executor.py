# stage_executor.py
import random
import time
from pipeline_stages import STAGES

class StageExecutor:
    def __init__(self):
        self.results = []

    def execute_stage(self, stage) -> dict:
        """执行单个阶段，返回结果"""
        time.sleep(random.uniform(0.3, 0.8))  # 模拟执行时间
        success_rates = {"lint": 0.95, "security": 0.90, "test": 0.85,
                         "build": 0.92, "deploy": 0.95, "verify": 0.98}
        rate = success_rates.get(stage.category, 0.9)
        success = random.random() < rate

        result = {"stage": stage.name, "status": "success" if success else "failure"}
        self.results.append(result)
        icon = "✅" if success else "❌"
        print(f"  {icon} {stage.name}")
        return result

    def execute_pipeline(self) -> dict:
        """执行完整流水线"""
        print("开始流水线...")
        for stage in STAGES:
            result = self.execute_stage(stage)
            if result["status"] == "failure" and stage.blocking:
                print(f"  🛑 阻断：{stage.name} 失败，跳过后续阶段")
                # 标记剩余阶段为 skipped
                remaining = [s for s in STAGES if STAGES.index(s) > STAGES.index(stage)]
                for s in remaining:
                    self.results.append({"stage": s.name, "status": "skipped"})
                    print(f"  ⏭️  {s.name} (skipped)")
                break

        passed = sum(1 for r in self.results if r["status"] == "success")
        failed = sum(1 for r in self.results if r["status"] == "failure")
        skipped = sum(1 for r in self.results if r["status"] == "skipped")
        print(f"\n结果: 通过={passed} 失败={failed} 跳过={skipped}")
        return {"passed": passed, "failed": failed, "skipped": skipped}

if __name__ == "__main__":
    executor = StageExecutor()
    result = executor.execute_pipeline()

