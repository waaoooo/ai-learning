# delivery_checklist.py
from datetime import datetime

class DeliveryChecklist:
    """企业级项目交付检查清单"""

    def __init__(self, project_name: str):
        self.project = project_name
        self.items = []

    def add(self, category: str, item: str, status: bool = False, note: str = ""):
        self.items.append({
            "category": category,
            "item": item,
            "status": status,
            "note": note,
        })

    def render(self) -> str:
        lines = []
        lines.append(f"# {self.project} — 交付清单")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 按类别分组
        by_cat = {}
        for item in self.items:
            by_cat.setdefault(item["category"], []).append(item)

        total = len(self.items)
        done = sum(1 for i in self.items if i["status"])
        lines.append(f"\n完成度: {done}/{total} ({done/total*100:.0f}%)")

        for cat, items in by_cat.items():
            cat_done = sum(1 for i in items if i["status"])
            lines.append(f"\n## {cat} ({cat_done}/{len(items)})")
            for item in items:
                icon = "✅" if item["status"] else "⬜"
                note = f" — {item['note']}" if item["note"] else ""
                lines.append(f"  {icon} {item['item']}{note}")

        return "\n".join(lines)

    def is_ready(self) -> bool:
        """是否可以交付"""
        critical = [i for i in self.items if i["category"] in ("代码", "测试", "安全")]
        return all(i["status"] for i in critical)


if __name__ == "__main__":
    checklist = DeliveryChecklist("AI 客服 Agent 平台")

    # 代码
    checklist.add("代码", "LLM 客户端封装", True, "重试 + 超时 + Token 统计")
    checklist.add("代码", "工具执行器", True, "注册 + 校验 + 错误处理")
    checklist.add("代码", "RAG 引擎", True, "切片 + 检索 + 生成")
    checklist.add("代码", "Agent 编排器", True, "路由 + 调度 + 上下文")
    checklist.add("代码", "多 Agent 协作", True, "客服 + 知识 + 工具")

    # 测试
    checklist.add("测试", "单元测试", True, "Mock LLM")
    checklist.add("测试", "集成测试", True, "Agent + 工具")
    checklist.add("测试", "E2E 测试", True, "完整链路")
    checklist.add("测试", "评测框架", True, "数据集 + 评分器 + 回归")

    # 安全
    checklist.add("安全", "输入过滤", True, "Prompt Injection 防御")
    checklist.add("安全", "RBAC 权限", True, "角色 + 工具白名单")
    checklist.add("安全", "内容审核", True, "敏感词 + PII + 合规")

    # 部署
    checklist.add("部署", "Docker 镜像", False, "待构建")
    checklist.add("部署", "K8s 配置", False, "待编写")
    checklist.add("部署", "CI/CD 流水线", True, "12 阶段")
    checklist.add("部署", "健康检查", True, "Liveness + Readiness")

    # 监控
    checklist.add("监控", "指标收集", True, "Prometheus")
    checklist.add("监控", "日志聚合", True, "结构化日志")
    checklist.add("监控", "告警规则", True, "5 条规则")
    checklist.add("监控", "仪表盘", True, "Grafana")

    # 文档
    checklist.add("文档", "OpenAPI 规范", True, "7 个端点")
    checklist.add("文档", "架构文档", True, "ADR + 数据流")
    checklist.add("文档", "运维手册", True, "部署 + 故障处理")
    checklist.add("文档", "API 文档", True, "端点 + 错误码")

    print(checklist.render())
    print(f"\n可交付: {'✅ 是' if checklist.is_ready() else '❌ 否'}")
