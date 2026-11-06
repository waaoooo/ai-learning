# project_skeleton.py
import os
import json
from datetime import datetime

class ProjectSkeleton:
    """企业级 AI 项目骨架"""

    def __init__(self, name: str):
        self.name = name
        self.modules = {}
        self.dependencies = {}

    def add_module(self, name: str, path: str, description: str, week: str):
        self.modules[name] = {
            "path": path,
            "description": description,
            "week": week,
            "status": "pending",
        }

    def add_dependency(self, from_mod: str, to_mod: str):
        self.dependencies.setdefault(from_mod, []).append(to_mod)

    def render(self) -> str:
        lines = []
        lines.append(f"# {self.name} — 项目骨架")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n## 模块清单\n")
        lines.append("| 模块 | 路径 | 描述 | 来源 | 状态 |")
        lines.append("|---|---|---|---|---|")
        for name, info in self.modules.items():
            lines.append(f"| {name} | {info['path']} | {info['description']} | {info['week']} | {info['status']} |")

        lines.append("\n## 依赖关系\n")
        for from_mod, to_mods in self.dependencies.items():
            for to_mod in to_mods:
                lines.append(f"- {from_mod} → {to_mod}")

        return "\n".join(lines)

if __name__ == "__main__":
    project = ProjectSkeleton("AI 客服 Agent 平台")

    # 模块清单（映射到 12 周学习内容）
    project.add_module("llm_client", "src/llm/", "LLM 调用封装", "W1")
    project.add_module("rag_engine", "src/rag/", "RAG 检索引擎", "W2")
    project.add_module("tool_executor", "src/tools/", "工具调用执行器", "W3")
    project.add_module("planner", "src/agent/planner.py", "任务规划器", "W4")
    project.add_module("memory_store", "src/memory/", "记忆存储", "W5")
    project.add_module("multi_agent", "src/agents/", "多 Agent 协作", "W6")
    project.add_module("eval_engine", "src/eval/", "评测引擎", "W7/W11")
    project.add_module("observability", "src/obs/", "可观测性", "W8/W11")
    project.add_module("security", "src/security/", "安全网关", "W9")
    project.add_module("infra", "src/infra/", "基础设施（路由/缓存/DB）", "W10")
    project.add_module("cicd", ".github/workflows/", "CI/CD 流水线", "W11")
    project.add_module("docs", "docs/", "文档与 API 规范", "W11")

    # 依赖关系
    project.add_dependency("planner", "llm_client")
    project.add_dependency("planner", "tool_executor")
    project.add_dependency("multi_agent", "planner")
    project.add_dependency("multi_agent", "memory_store")
    project.add_dependency("eval_engine", "multi_agent")
    project.add_dependency("observability", "eval_engine")
    project.add_dependency("security", "observability")
    project.add_dependency("infra", "security")
    project.add_dependency("cicd", "infra")
    project.add_dependency("docs", "cicd")

    print(project.render())
