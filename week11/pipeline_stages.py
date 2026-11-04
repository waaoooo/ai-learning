# pipeline_stages.py
from dataclasses import dataclass

@dataclass
class PipelineStage:
    name: str
    category: str       # lint/test/build/deploy/verify/security
    blocking: bool      # 是否阻断后续阶段
    timeout: int        # 超时秒数

STAGES = [
    PipelineStage("lint", "lint", True, 120),
    PipelineStage("security_scan", "security", True, 180),
    PipelineStage("unit_test", "test", True, 300),
    PipelineStage("integration_test", "test", True, 600),
    PipelineStage("eval_regression", "test", True, 900),   # ← AI 特有
    PipelineStage("build", "build", True, 600),
    PipelineStage("image_scan", "security", True, 300),
    PipelineStage("deploy_staging", "deploy", True, 300),
    PipelineStage("smoke_test", "test", True, 120),
    PipelineStage("perf_benchmark", "verify", False, 600),  # 非阻断
    PipelineStage("deploy_prod", "deploy", True, 300),
    PipelineStage("verify_prod", "verify", True, 120),
]

if __name__ == "__main__":
    print(f"共 {len(STAGES)} 个阶段:")
    for i, s in enumerate(STAGES, 1):
        blocking = "🔴 阻断" if s.blocking else "🟡 非阻断"
        print(f"  {i:2d}. {s.name:20s} [{s.category:8s}] {blocking}")

