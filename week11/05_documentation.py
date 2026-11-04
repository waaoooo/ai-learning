"""
W11 第 5 课：文档与 API 规范

演示：
  ① OpenAPI 规范自动生成
  ② 架构文档（组件图 + 数据流）
  ③ API 文档（端点 + 请求/响应）
  ④ 运维手册（部署 + 监控 + 故障处理）
  ⑤ 文档版本管理
  ⑥ 文档质量检查
"""
import os
import json
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 0.5


# ══════════════════════════════════════════════════════════
# ① OpenAPI 规范自动生成
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① OpenAPI 规范自动生成")
print("=" * 60)

class OpenAPIGenerator:
    """OpenAPI 3.0 规范生成器"""

    def __init__(self, title: str, version: str, base_url: str):
        self.spec = {
            "openapi": "3.0.0",
            "info": {"title": title, "version": version, "description": f"{title} API"},
            "servers": [{"url": base_url}],
            "paths": {},
            "components": {"schemas": {}},
        }

    def add_endpoint(self, path: str, method: str, summary: str,
                     request_schema: dict = None, response_schema: dict = None,
                     tags: list = None):
        """添加端点"""
        if path not in self.spec["paths"]:
            self.spec["paths"][path] = {}

        endpoint = {
            "summary": summary,
            "tags": tags or ["default"],
            "responses": {"200": {"description": "成功"}},
        }

        if request_schema:
            endpoint["requestBody"] = {
                "required": True,
                "content": {"application/json": {"schema": request_schema}}
            }
        if response_schema:
            endpoint["responses"]["200"]["content"] = {
                "application/json": {"schema": response_schema}
            }

        self.spec["paths"][path][method.lower()] = endpoint

    def add_schema(self, name: str, schema: dict):
        self.spec["components"]["schemas"][name] = schema

    def export_json(self) -> str:
        return json.dumps(self.spec, indent=2, ensure_ascii=False)

    def export_yaml(self) -> str:
        # 简化 YAML 导出
        lines = []
        lines.append(f"openapi: 3.0.0")
        lines.append(f"info:")
        lines.append(f"  title: {self.spec['info']['title']}")
        lines.append(f"  version: {self.spec['info']['version']}")
        lines.append(f"servers:")
        for s in self.spec["servers"]:
            lines.append(f"  - url: {s['url']}")
        lines.append(f"paths:")
        for path, methods in self.spec["paths"].items():
            lines.append(f"  {path}:")
            for method, info in methods.items():
                lines.append(f"    {method}:")
                lines.append(f"      summary: {info['summary']}")
                lines.append(f"      tags: {info['tags']}")
        return "\n".join(lines)


api = OpenAPIGenerator("Agent Platform API", "1.0.0", "https://api.example.com/v1")

# 定义 Schema
api.add_schema("ChatRequest", {
    "type": "object",
    "properties": {
        "message": {"type": "string", "description": "用户消息"},
        "session_id": {"type": "string", "description": "会话 ID"},
        "temperature": {"type": "number", "default": 0.7},
    },
    "required": ["message"],
})

api.add_schema("ChatResponse", {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "tokens_used": {"type": "integer"},
        "model": {"type": "string"},
        "latency_ms": {"type": "number"},
    },
})

# 添加端点
api.add_endpoint("/chat", "POST", "发送消息给 Agent", 
    request_schema={"$ref": "#/components/schemas/ChatRequest"},
    response_schema={"$ref": "#/components/schemas/ChatResponse"},
    tags=["chat"])

api.add_endpoint("/chat/stream", "POST", "流式聊天", tags=["chat"])
api.add_endpoint("/sessions", "GET", "列出会话", tags=["sessions"])
api.add_endpoint("/sessions/{id}", "GET", "获取会话详情", tags=["sessions"])
api.add_endpoint("/sessions/{id}", "DELETE", "删除会话", tags=["sessions"])
api.add_endpoint("/health", "GET", "健康检查", tags=["system"])
api.add_endpoint("/metrics", "GET", "Prometheus 指标", tags=["system"])

spec = api.spec
print(f"  OpenAPI 版本: {spec['openapi']}")
print(f"  端点数量: {sum(len(v) for v in spec['paths'].values())}")
print(f"  Schema 数量: {len(spec['components']['schemas'])}")
print(f"\n  端点列表:")
for path, methods in spec["paths"].items():
    for method in methods:
        print(f"    {method.upper():6s} {path}")

print(f"\n  YAML 导出 (节选):")
for line in api.export_yaml().split("\n")[:15]:
    print(f"    {line}")


# ══════════════════════════════════════════════════════════
# ② 架构文档
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 架构文档")
print("=" * 60)

class ArchitectureDoc:
    """架构文档生成器"""

    def __init__(self):
        self.components = []
        self.data_flows = []
        self.decisions = []

    def add_component(self, name: str, type: str, description: str, tech: str):
        self.components.append({
            "name": name, "type": type, "description": description, "tech": tech
        })

    def add_data_flow(self, from_comp: str, to_comp: str, data: str, protocol: str):
        self.data_flows.append({
            "from": from_comp, "to": to_comp, "data": data, "protocol": protocol
        })

    def add_decision(self, title: str, context: str, decision: str, consequences: str):
        self.decisions.append({
            "title": title, "context": context, "decision": decision, "consequences": consequences
        })

    def render_markdown(self) -> str:
        lines = []
        lines.append("# 系统架构文档")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n## 1. 组件清单")
        lines.append("\n| 组件 | 类型 | 描述 | 技术栈 |")
        lines.append("|---|---|---|---|")
        for c in self.components:
            lines.append(f"| {c['name']} | {c['type']} | {c['description']} | {c['tech']} |")

        lines.append("\n## 2. 数据流")
        for flow in self.data_flows:
            lines.append(f"- **{flow['from']}** → **{flow['to']}**: {flow['data']} ({flow['protocol']})")

        lines.append("\n## 3. 架构决策记录 (ADR)")
        for i, d in enumerate(self.decisions, 1):
            lines.append(f"\n### ADR-{i}: {d['title']}")
            lines.append(f"\n**背景**: {d['context']}")
            lines.append(f"\n**决策**: {d['decision']}")
            lines.append(f"\n**后果**: {d['consequences']}")

        return "\n".join(lines)


arch = ArchitectureDoc()

# 组件
arch.add_component("API Gateway", "gateway", "路由、限流、认证", "FastAPI + Nginx")
arch.add_component("Agent Orchestrator", "service", "多 Agent 调度", "Python + asyncio")
arch.add_component("LLM Proxy", "service", "模型路由 + 缓存", "Python + Redis")
arch.add_component("Vector DB", "database", "向量存储 + 检索", "pgvector / Milvus")
arch.add_component("Message Queue", "infra", "异步消息", "Redis Streams")
arch.add_component("Monitoring", "infra", "指标 + 日志 + 追踪", "Prometheus + Grafana")

# 数据流
arch.add_data_flow("Client", "API Gateway", "HTTP 请求", "REST")
arch.add_data_flow("API Gateway", "Agent Orchestrator", "路由请求", "gRPC")
arch.add_data_flow("Agent Orchestrator", "LLM Proxy", "LLM 调用", "HTTP")
arch.add_data_flow("LLM Proxy", "Vector DB", "向量检索", "SQL")
arch.add_data_flow("Agent Orchestrator", "Message Queue", "异步任务", "Redis")
arch.add_data_flow("All Services", "Monitoring", "指标/日志", "Prometheus")

# ADR
arch.add_decision("模型路由策略", "需要在成本和质量间平衡",
                  "按任务复杂度路由：simple→flash, complex→plus",
                  "成本降低 60%，但需要维护路由规则")
arch.add_decision("缓存策略", "重复请求浪费 Token",
                  "语义缓存（相似度 > 0.95 复用）",
                  "缓存命中率 40%，节省 30% Token")

doc = arch.render_markdown()
print(doc[:800])
print("  ... (完整文档已生成)")


# ══════════════════════════════════════════════════════════
# ③ API 文档
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ API 文档")
print("=" * 60)

class APIDocGenerator:
    """API 文档生成器"""

    def __init__(self):
        self.endpoints = []

    def add(self, method: str, path: str, description: str,
            request: dict = None, response: dict = None, errors: list = None):
        self.endpoints.append({
            "method": method, "path": path, "description": description,
            "request": request, "response": response, "errors": errors or [],
        })

    def render_markdown(self) -> str:
        lines = []
        lines.append("# API 文档")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n## 端点列表\n")

        for ep in self.endpoints:
            lines.append(f"### {ep['method']} {ep['path']}")
            lines.append(f"\n{ep['description']}\n")

            if ep["request"]:
                lines.append("**请求体:**")
                lines.append("```json")
                lines.append(json.dumps(ep["request"], indent=2, ensure_ascii=False))
                lines.append("```")

            if ep["response"]:
                lines.append("**响应:**")
                lines.append("```json")
                lines.append(json.dumps(ep["response"], indent=2, ensure_ascii=False))
                lines.append("```")

            if ep["errors"]:
                lines.append("**错误码:**")
                for err in ep["errors"]:
                    lines.append(f"- `{err['code']}`: {err['message']}")

            lines.append("")

        return "\n".join(lines)


doc_gen = APIDocGenerator()

doc_gen.add("POST", "/chat", "发送消息给 Agent",
    request={"message": "你好", "session_id": "sess-001", "temperature": 0.7},
    response={"reply": "您好！有什么可以帮助您的？", "tokens_used": 150, "model": "glm-4-flash", "latency_ms": 1234.5},
    errors=[{"code": 400, "message": "message 不能为空"}, {"code": 429, "message": "请求过于频繁"}, {"code": 500, "message": "内部错误"}])

doc_gen.add("GET", "/sessions", "列出所有会话",
    response={"sessions": [{"id": "sess-001", "created_at": "2026-11-02T10:00:00Z", "message_count": 5}]},
    errors=[{"code": 401, "message": "未授权"}])

doc_gen.add("DELETE", "/sessions/{id}", "删除会话",
    response={"deleted": True},
    errors=[{"code": 404, "message": "会话不存在"}])

api_doc = doc_gen.render_markdown()
print(api_doc[:600])
print("  ... (完整文档已生成)")


# ══════════════════════════════════════════════════════════
# ④ 运维手册
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 运维手册")
print("=" * 60)

class OpsManual:
    """运维手册生成器"""

    def __init__(self):
        self.sections = {}

    def add_section(self, title: str, content: str):
        self.sections[title] = content

    def render(self) -> str:
        lines = []
        lines.append("# 运维手册")
        lines.append(f"\n版本: 1.0.0 | 更新: {datetime.now().strftime('%Y-%m-%d')}")
        for title, content in self.sections.items():
            lines.append(f"\n## {title}")
            lines.append(content)
        return "\n".join(lines)


ops = OpsManual()

ops.add_section("部署流程", """
### 部署步骤
1. 构建镜像: `docker build -t agent:v1.0 .`
2. 推送镜像: `docker push registry.example.com/agent:v1.0`
3. 更新 K8s: `kubectl set image deployment/agent agent=registry.example.com/agent:v1.0`
4. 验证: `kubectl rollout status deployment/agent`
5. 回滚: `kubectl rollout undo deployment/agent`

### 环境变量
| 变量 | 说明 | 示例 |
|---|---|---|
| ZHIPU_API_KEY | API 密钥 | sk-xxx |
| DATABASE_URL | 数据库连接 | postgresql://... |
| REDIS_URL | Redis 连接 | redis://... |
""")

ops.add_section("监控检查", """
### 健康检查
- 存活: `GET /health` → 200 OK
- 就绪: `GET /ready` → 200 OK (所有依赖正常)

### 关键指标
| 指标 | 阈值 | 告警级别 |
|---|---|---|
| 错误率 | >5% | Critical |
| P95 延迟 | >10s | Critical |
| 缓存命中率 | <30% | Warning |
| Token 用量 | >2000/请求 | Warning |

### 日志查询
```bash
# 查看 ERROR 日志
kubectl logs -l app=agent | jq 'select(.level == "ERROR")'

# 查看特定请求
kubectl logs -l app=agent | jq 'select(.request_id == "req-001")'
```
""")

ops.add_section("故障处理", """
### 常见故障

**1. LLM API 超时**
- 症状: 延迟 >30s, 错误率上升
- 处理: 切换备用模型，启用降级
- 命令: `kubectl set env deployment/agent FALLBACK_MODEL=glm-4-flash`

**2. 数据库连接失败**
- 症状: 500 错误，日志显示 connection refused
- 处理: 检查数据库 Pod，重启连接池
- 命令: `kubectl rollout restart deployment/agent`

**3. 内存泄漏**
- 症状: 内存持续增长，OOM Kill
- 处理: 重启 Pod，分析 heap dump
- 命令: `kubectl delete pod -l app=agent`

### 升级流程
1. 确认故障影响范围
2. 通知相关人员
3. 执行回滚或修复
4. 验证恢复
5. 编写事故报告
""")

manual = ops.render()
print(manual[:700])
print("  ... (完整手册已生成)")


# ══════════════════════════════════════════════════════════
# ⑤ 文档版本管理
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑤ 文档版本管理")
print("=" * 60)

class DocVersionManager:
    """文档版本管理器"""

    def __init__(self):
        self.versions = {}  # {doc_name: {version: content}}

    def save(self, doc_name: str, version: str, content: str, author: str = "system"):
        if doc_name not in self.versions:
            self.versions[doc_name] = {}
        self.versions[doc_name][version] = {
            "content": content,
            "author": author,
            "timestamp": datetime.now().isoformat(),
        }
        print(f"  📄 保存: {doc_name} v{version} (by {author})")

    def get(self, doc_name: str, version: str = None) -> str:
        if doc_name not in self.versions:
            return None
        if version:
            return self.versions[doc_name].get(version, {}).get("content")
        # 返回最新版本
        versions = sorted(self.versions[doc_name].keys())
        return self.versions[doc_name][versions[-1]]["content"] if versions else None

    def diff(self, doc_name: str, v1: str, v2: str) -> str:
        c1 = self.get(doc_name, v1) or ""
        c2 = self.get(doc_name, v2) or ""
        lines1 = c1.split("\n")
        lines2 = c2.split("\n")
        diff_lines = []
        for i, (a, b) in enumerate(zip(lines1, lines2)):
            if a != b:
                diff_lines.append(f"  L{i+1}: - {a}")
                diff_lines.append(f"  L{i+1}: + {b}")
        return "\n".join(diff_lines) if diff_lines else "  无差异"

    def list_versions(self, doc_name: str) -> list:
        if doc_name not in self.versions:
            return []
        return sorted(self.versions[doc_name].keys())


vm = DocVersionManager()

# 保存多个版本
vm.save("api-doc", "1.0", "# API v1.0\n端点: /chat", "alice")
time.sleep(0.1)
vm.save("api-doc", "1.1", "# API v1.1\n端点: /chat, /sessions", "bob")
time.sleep(0.1)
vm.save("api-doc", "2.0", "# API v2.0\n端点: /chat, /sessions, /health", "alice")

print(f"\n  版本列表: {vm.list_versions('api-doc')}")
print(f"\n  v1.0 → v1.1 差异:")
print(vm.diff("api-doc", "1.0", "1.1"))


# ══════════════════════════════════════════════════════════
# ⑥ 文档质量检查
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑥ 文档质量检查")
print("=" * 60)

class DocQualityChecker:
    """文档质量检查器"""

    CHECKS = {
        "has_title": lambda doc: "# " in doc,
        "has_description": lambda doc: len(doc) > 100,
        "has_examples": lambda doc: "```" in doc,
        "has_error_codes": lambda doc: "错误" in doc or "error" in doc.lower(),
        "has_version": lambda doc: "版本" in doc or "version" in doc.lower(),
        "no_todo": lambda doc: "TODO" not in doc and "FIXME" not in doc,
    }

    def check(self, doc: str, doc_name: str = "document") -> dict:
        results = {}
        for name, check_fn in self.CHECKS.items():
            try:
                results[name] = check_fn(doc)
            except Exception:
                results[name] = False

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        print(f"  文档: {doc_name}")
        print(f"  质量得分: {passed}/{total} ({passed/total*100:.0f}%)")
        for name, result in results.items():
            icon = "✅" if result else "❌"
            print(f"    {icon} {name}")

        return {"passed": passed, "total": total, "score": passed / total, "details": results}


checker = DocQualityChecker()

# 检查 API 文档
checker.check(api_doc, "API 文档")

# 检查运维手册
checker.check(manual, "运维手册")

# 检查架构文档
checker.check(doc, "架构文档")


# ══════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("W11 第 5 课总结")
print("=" * 60)
print("""
  ① OpenAPI 规范：自动生成 API 文档，支持 JSON/YAML 导出
  ② 架构文档：组件清单 + 数据流 + ADR（架构决策记录）
  ③ API 文档：端点 + 请求/响应 + 错误码
  ④ 运维手册：部署流程 + 监控检查 + 故障处理
  ⑤ 文档版本管理：版本控制 + diff 对比
  ⑥ 文档质量检查：6 项检查（标题/描述/示例/错误码/版本/无 TODO）

  文档是系统的"说明书" —— 好的文档让新人 1 天上手
  API 文档自动生成：代码即文档，避免文档过时
  运维手册：故障时的"救命手册"
""")
print("=" * 60)
print("\n  问你一个问题（引向 W12）：")
print("  企业级项目实战应该包含哪些环节？")
print("  （需求分析、架构设计、开发、测试、部署、运维……）")
print("=" * 60)
