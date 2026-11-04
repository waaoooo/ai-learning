"""
W11 第 3 课：CI/CD 流水线

演示：
  ① 流水线阶段定义（12 个阶段）
  ② 阶段执行器（顺序/并行/条件执行）
  ③ 门禁检查（Gate Check）
  ④ 自动回滚机制
  ⑤ 流水线报告
"""
import os
import json
import time
import random
import subprocess
from datetime import datetime
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
# ① 流水线阶段定义
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① CI/CD 流水线阶段定义")
print("=" * 60)

@dataclass
class PipelineStage:
    name: str
    description: str
    category: str  # lint/test/build/deploy/verify
    blocking: bool = True  # 是否阻断后续阶段
    timeout: int = 300  # 超时秒数
    retry: int = 0  # 重试次数


from dataclasses import dataclass, field

stages = [
    PipelineStage("lint", "代码风格 + 类型检查", "lint", True, 120, 0),
    PipelineStage("security_scan", "安全扫描 + 依赖漏洞", "security", True, 180, 0),
    PipelineStage("unit_test", "单元测试", "test", True, 300, 1),
    PipelineStage("integration_test", "集成测试", "test", True, 600, 1),
    PipelineStage("eval_regression", "评测回归测试", "test", True, 900, 0),
    PipelineStage("build", "Docker 镜像构建", "build", True, 600, 1),
    PipelineStage("image_scan", "镜像安全扫描", "security", True, 300, 0),
    PipelineStage("deploy_staging", "部署到 Staging", "deploy", True, 300, 1),
    PipelineStage("smoke_test", "冒烟测试 (Staging)", "test", True, 120, 1),
    PipelineStage("perf_benchmark", "性能基准测试", "verify", False, 600, 0),
    PipelineStage("deploy_prod", "部署到 Production", "deploy", True, 300, 1),
    PipelineStage("verify_prod", "生产验证", "verify", True, 120, 1),
]

print(f"  共 {len(stages)} 个阶段:")
for i, s in enumerate(stages, 1):
    blocking = "🔴 阻断" if s.blocking else "🟡 非阻断"
    print(f"    {i:2d}. {s.name:20s} [{s.category:8s}] {blocking} timeout={s.timeout}s")


# ══════════════════════════════════════════════════════════
# ② 阶段执行器
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 阶段执行器")
print("=" * 60)

class StageExecutor:
    """阶段执行器：模拟 CI/CD 阶段执行"""

    def __init__(self):
        self.results = []
        self.start_time = time.time()

    def execute_stage(self, stage: PipelineStage) -> dict:
        """执行单个阶段"""
        start = time.time()
        print(f"  ▶ 执行: {stage.name}...", end=" ", flush=True)

        # 模拟执行时间
        exec_time = random.uniform(0.5, 2.0)
        time.sleep(exec_time)

        # 模拟成功率（不同阶段不同成功率）
        success_rates = {
            "lint": 0.95, "security": 0.90, "test": 0.85,
            "build": 0.92, "deploy": 0.95, "verify": 0.98,
        }
        rate = success_rates.get(stage.category, 0.9)
        success = random.random() < rate

        # 模拟特定失败
        if stage.name == "eval_regression":
            # 评测回归：模拟 10% 概率失败
            success = random.random() > 0.1

        result = {
            "stage": stage.name,
            "status": "success" if success else "failure",
            "duration": round(time.time() - start, 2),
            "timestamp": datetime.now().isoformat(),
            "logs": f"[{stage.name}] {'PASSED' if success else 'FAILED'}",
        }

        icon = "✅" if success else "❌"
        print(f"{icon} ({result['duration']}s)")

        self.results.append(result)
        return result

    def execute_pipeline(self, stages: list[PipelineStage]) -> dict:
        """执行完整流水线"""
        print(f"\n  开始流水线执行 ({len(stages)} 个阶段)...\n")

        for stage in stages:
            result = self.execute_stage(stage)

            if result["status"] == "failure" and stage.blocking:
                print(f"\n  🛑 阻断：{stage.name} 失败，停止后续阶段")
                # 记录跳过的阶段
                remaining = stages[stages.index(stage) + 1:]
                for s in remaining:
                    self.results.append({
                        "stage": s.name,
                        "status": "skipped",
                        "duration": 0,
                        "timestamp": datetime.now().isoformat(),
                        "logs": "跳过（前置阶段失败）",
                    })
                break

        total_time = time.time() - self.start_time
        passed = sum(1 for r in self.results if r["status"] == "success")
        failed = sum(1 for r in self.results if r["status"] == "failure")
        skipped = sum(1 for r in self.results if r["status"] == "skipped")

        return {
            "total_stages": len(stages),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total_time": round(total_time, 2),
            "status": "success" if failed == 0 else "failure",
        }


executor = StageExecutor()
pipeline_result = executor.execute_pipeline(stages)

print(f"\n  流水线结果: {pipeline_result['status'].upper()}")
print(f"  通过: {pipeline_result['passed']}, 失败: {pipeline_result['failed']}, 跳过: {pipeline_result['skipped']}")
print(f"  总耗时: {pipeline_result['total_time']}s")


# ══════════════════════════════════════════════════════════
# ③ 门禁检查 (Gate Check)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 门禁检查 (Gate Check)")
print("=" * 60)

class GateChecker:
    """门禁检查：控制是否允许进入下一阶段"""

    GATES = {
        "lint_to_test": lambda r: r.get("lint", {}).get("status") == "success",
        "test_to_build": lambda r: all(r.get(t, {}).get("status") == "success" for t in ["unit_test", "integration_test"]),
        "build_to_deploy": lambda r: r.get("image_scan", {}).get("status") == "success",
        "staging_to_prod": lambda r: r.get("smoke_test", {}).get("status") == "success",
    }

    def check(self, gate_name: str, results: dict) -> tuple[bool, str]:
        gate = self.GATES.get(gate_name)
        if not gate:
            return True, "无门禁规则"
        try:
            passed = gate(results)
            return passed, "通过" if passed else "阻断"
        except Exception as e:
            return False, f"门禁检查异常: {e}"


gate = GateChecker()

# 构建结果字典
results_dict = {r["stage"]: r for r in executor.results}

print("  门禁检查:")
for gate_name in GateChecker.GATES:
    passed, msg = gate.check(gate_name, results_dict)
    icon = "✅" if passed else "❌"
    print(f"    {icon} {gate_name}: {msg}")


# ══════════════════════════════════════════════════════════
# ④ 自动回滚机制
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 自动回滚机制")
print("=" * 60)

class RollbackManager:
    """自动回滚管理器"""

    def __init__(self):
        self.deployment_history = []
        self.current_version = "v1.0.0"

    def deploy(self, version: str) -> bool:
        """部署新版本"""
        print(f"  📦 部署 {version}...")
        self.deployment_history.append({"version": version, "status": "deploying", "time": time.time()})
        # 模拟部署
        time.sleep(0.5)
        success = random.random() > 0.2  # 80% 成功率
        if success:
            self.current_version = version
            self.deployment_history[-1]["status"] = "success"
            print(f"  ✅ 部署成功: {version}")
        else:
            self.deployment_history[-1]["status"] = "failed"
            print(f"  ❌ 部署失败: {version}")
            self.rollback()
        return success

    def rollback(self):
        """回滚到上一版本"""
        if len(self.deployment_history) < 2:
            print("  ⚠️ 无历史版本可回滚")
            return
        # 找到上一个成功版本
        for entry in reversed(self.deployment_history[:-1]):
            if entry["status"] == "success":
                self.current_version = entry["version"]
                print(f"  🔄 回滚到: {entry['version']}")
                return
        print("  ⚠️ 无成功版本可回滚")

    def status(self):
        print(f"  当前版本: {self.current_version}")
        print(f"  部署历史: {len(self.deployment_history)} 次")


rb = RollbackManager()
print("  初始状态:")
rb.status()

# 模拟多次部署
for v in ["v1.1.0", "v1.2.0", "v1.3.0"]:
    print(f"\n  部署 {v}:")
    rb.deploy(v)
    time.sleep(0.3)

print("\n  最终状态:")
rb.status()


# ══════════════════════════════════════════════════════════
# ⑤ 流水线报告
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑤ 流水线报告")
print("=" * 60)

class PipelineReport:
    """流水线报告生成器"""

    @staticmethod
    def generate(results: list[dict], pipeline_result: dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("CI/CD 流水线报告")
        lines.append("=" * 60)
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"状态: {pipeline_result['status'].upper()}")
        lines.append(f"总阶段: {pipeline_result['total_stages']}")
        lines.append(f"通过: {pipeline_result['passed']}, 失败: {pipeline_result['failed']}, 跳过: {pipeline_result['skipped']}")
        lines.append(f"总耗时: {pipeline_result['total_time']}s")

        lines.append(f"\n--- 阶段详情 ---")
        for r in results:
            icon = {"success": "✅", "failure": "❌", "skipped": "⏭️"}.get(r["status"], "❓")
            lines.append(f"  {icon} {r['stage']:20s} {r['status']:8s} {r['duration']:6.2f}s")

        # 失败分析
        failures = [r for r in results if r["status"] == "failure"]
        if failures:
            lines.append(f"\n--- 失败分析 ---")
            for f in failures:
                lines.append(f"  ❌ {f['stage']}: {f['logs']}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


report = PipelineReport.generate(executor.results, pipeline_result)
print(report)

# ══════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("W11 第 3 课总结")
print("=" * 60)
print("""
  ① 流水线阶段：12 个阶段覆盖 Lint → Security → Test → Build → Deploy → Verify
  ② 门禁检查：阶段间自动门禁，阻断不合格代码
  ③ 自动回滚：部署失败自动回滚到上一版本
  ④ 流水线报告：通过/失败/跳过统计 + 耗时分析

  CI/CD 是 Agent 系统的"质量门" —— 每次提交自动验证
  关键：Eval Regression 是 AI 系统特有的阶段
""")
print("=" * 60)
print("\n  问你一个问题（引向 W11 第 4 课）：")
print("  监控与告警系统应该包含哪些组件？")
print("  （指标收集、日志聚合、追踪、告警规则……）")
print("=" * 60)
