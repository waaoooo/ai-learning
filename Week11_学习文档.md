# Week 11 学习文档：工程化与评测

> 完成日期：2026-11-02
> 脚本目录：`week11/`
> 状态：✅ 已验收

---

## 一、本周概览

Week 11 聚焦 AI 系统的**工程化**能力：测试、评测、CI/CD、监控、文档。与 Week 7（评测方法）和 Week 8（可观测性）不同，Week 11 关注的是**工程实践**——如何把 AI 系统变成可维护、可部署、可监控的生产系统。

### 5 个课时

| 课时 | 主题 | 核心概念 |
|---|---|---|
| T1 | 测试体系 | Mock LLM + 单元/集成/E2E 测试 |
| T2 | 评测框架 | 评测数据集 + 3 种评分器 + 回归测试 |
| T3 | CI/CD 流水线 | 12 阶段 + 门禁 + 自动回滚 |
| T4 | 监控与告警 | 指标 + 日志 + 追踪 + 告警 + 自愈 |
| T5 | 文档与 API 规范 | OpenAPI + 架构文档 + 质量检查 |

---

## 二、T1：测试体系

### 核心概念

**Mock LLM**：用预设响应替代真实 LLM 调用，让测试可控、可重复。

**测试金字塔**：
- 单元测试（多）：单个函数/类，Mock LLM
- 集成测试（中）：组件协作（Agent + 工具）
- E2E 测试（少）：完整链路

### 关键代码

```python
class MockLLM:
    def __init__(self, preset_responses: dict):
        self.preset = preset_responses
        self.call_count = 0

    def chat(self, messages: list) -> str:
        self.call_count += 1
        full_text = " ".join(m.get("content", "") for m in messages)
        for keyword, response in self.preset.items():
            if keyword in full_text:
                return response
        return "抱歉，我不知道"
```

### 关键教训

1. **Mock LLM 搜索所有消息**，不只是最后一条（系统消息中的关键词也要匹配）
2. **工具异常必须捕获** —— `try/except` 包裹工具调用，返回错误信息而非崩溃
3. **测试金字塔**：单元测试多、集成测试中、E2E 测试少

### 验收数据

- 13 个测试（7 单元 + 3 集成 + 3 E2E），覆盖率 93.9%

---

## 三、T2：评测框架

### 核心概念

**评测数据集**：结构化测试用例，包含 id/question/gold_answer/expected_keywords/difficulty/tags。

**三种评分器**：
- 精确匹配（Exact Match）：完全相同得 1.0
- 关键词匹配（Keyword Match）：命中率 = 命中数 / 总数
- LLM-as-Judge：语义评分 1-5 分

**回归测试**：保存基线，accuracy 下降 >2% 阻断发布。

### 关键代码

```python
class EvalRunner:
    def run(self) -> dict:
        for case in EVAL_DATASET:
            try:
                pred = self.agent_fn(case["question"])
                kw_score = keyword_match(pred, case["expected_keywords"])
                judge_result = llm_judge(pred, case["question"], case["gold_answer"])
                result = {"id": case["id"], "status": "pass" if kw_score >= 0.5 else "fail", ...}
            except Exception as e:
                result = {"id": case["id"], "status": "error", "error": str(e)}
            self.results.append(result)
        return self.summary()
```

### 关键教训

1. **评测集需要难度分级** —— simple/medium/complex，定位模型弱点
2. **评分器组合使用** —— 关键词做快速过滤，LLM 做深度评分
3. **错误隔离** —— 单条失败不中断整体评测
4. **回归阈值** —— accuracy 下降 >2% 阻断发布

### 验收数据

- 5 条测试用例（simple/medium/complex），平均关键词得分 0.933
- 回归检查正确检测 66.7% 下降 → REGRESSION

---

## 四、T3：CI/CD 流水线

### 核心概念

**12 阶段流水线**：Lint → Security → Test → Build → Deploy → Verify

**AI 特有阶段**：Eval Regression（评测回归）— 普通测试覆盖不了模型退化

**阻断机制**：阻断阶段失败 → 后续全部跳过（快速失败）

**门禁**：阶段间检查点，控制是否进入下一阶段

**自动回滚**：部署失败 → 自动回滚到上一成功版本

### 关键代码

```python
class StageExecutor:
    def execute_pipeline(self):
        for stage in STAGES:
            result = self.execute_stage(stage)
            if result["status"] == "failure" and stage.blocking:
                # 跳过后续阶段
                for s in remaining:
                    self.results.append({"stage": s.name, "status": "skipped"})
                break
```

### 关键教训

1. **Eval Regression 是 AI CI/CD 的核心** —— 模型更新后必须跑评测
2. **快速失败** —— 越早发现问题，修复成本越低
3. **自动回滚** —— 减少人工干预，缩短故障恢复时间
4. **非阻断阶段** —— perf_benchmark 失败不阻断发布

---

## 五、T4：监控与告警

### 核心概念

**三种指标类型**：
- Counter（计数器）：只增不减，请求总数
- Gauge（仪表盘）：可增可减，当前连接数
- Histogram（直方图）：分布统计，P50/P95/P99

**告警规则**：条件 + 阈值 + 严重级别 + 冷却时间

**健康检查**：Liveness（存活）+ Readiness（就绪）+ 自愈

### 关键代码

```python
class AlertEngine:
    def evaluate(self, metrics: dict) -> list:
        triggered = []
        for rule in self.rules:
            if rule.name in self.last_alert:
                if now - self.last_alert[rule.name] < rule.cooldown:
                    continue  # 冷却中，跳过
            if rule.condition(metrics):
                triggered.append(alert)
                self.last_alert[rule.name] = now
        return triggered
```

### 关键教训

1. **监控是系统的"眼睛"** —— 没有监控，你不知道系统是否健康
2. **告警冷却** —— 防止同一问题反复通知
3. **自愈减少人工** —— 简单故障自动恢复
4. **Liveness vs Readiness** —— 存活检查重启 Pod，就绪检查停止流量

---

## 六、T5：文档与 API 规范

### 核心概念

**OpenAPI**：REST API 标准描述格式，代码即文档

**架构文档**：组件清单 + 数据流 + ADR（架构决策记录）

**文档质量检查**：6 项自动检查（标题/描述/示例/错误码/版本/无 TODO）

### 关键代码

```python
class DocQualityChecker:
    CHECKS = {
        "has_title": lambda doc: "# " in doc,
        "has_description": lambda doc: len(doc) > 100,
        "has_examples": lambda doc: "```" in doc,
        "has_error_codes": lambda doc: "错误" in doc or "error" in doc.lower(),
        "has_version": lambda doc: "版本" in doc or "version" in doc.lower(),
        "no_todo": lambda doc: "TODO" not in doc and "FIXME" not in doc,
    }
```

### 关键教训

1. **文档即代码** —— 从代码注解自动生成，避免文档过时
2. **ADR 记录决策** —— 不仅记录"做了什么"，还记录"为什么这么做"
3. **文档质量可量化** —— 6 项检查，分数化评估

---

## 七、本周踩坑记录

| 坑 | 原因 | 修复 |
|---|---|---|
| MockLLM 只检查最后一条消息 | 忽略了系统消息中的关键词 | 搜索所有消息 `full_text = " ".join(...)` |
| 工具异常未捕获 | Agent 直接调用工具，异常传播 | `try/except` 包裹，返回错误信息 |
| f-string 反斜杠 | Python 3.9 不支持 f-string 中的反斜杠 | 提取变量后打印 |
| 嵌套 defaultdict 错误 | `defaultdict(list)` 不支持嵌套索引 | `defaultdict(lambda: defaultdict(list))` |
| 三引号字符串换行 | 代码块中的三引号字符串导致语法错误 | 用 `\n` 转义字符串 |

---

## 八、验收清单

- [x] T1：Mock LLM + 单元/集成/E2E 测试，13 个测试全部通过，覆盖率 93.9%
- [x] T2：评测数据集 + 3 种评分器 + 评测调度 + 报告 + 回归测试
- [x] T3：12 阶段 CI/CD + 门禁 + 自动回滚 + 流水线报告
- [x] T4：指标收集 + 日志聚合 + 链路追踪 + 告警引擎 + 健康检查 + 仪表盘
- [x] T5：OpenAPI + 架构文档 + API 文档 + 运维手册 + 版本管理 + 质量检查

---

## 九、W11 核心架构图

```
┌─────────────────────────────────────────────────────┐
│                  CI/CD 流水线                        │
│  Lint → Security → Unit Test → Integration Test     │
│  → Eval Regression → Build → Image Scan              │
│  → Deploy Staging → Smoke Test → Perf Benchmark      │
│  → Deploy Prod → Verify Prod                         │
├─────────────────────────────────────────────────────┤
│                  监控与告警                          │
│  Metrics (Counter/Gauge/Histogram)                   │
│  Logs (Structured + Indexed)                        │
│  Traces (Trace/Span Tree)                           │
│  Alerts (Rules + Cooldown + Notify)                 │
│  Health (Liveness/Readiness + Self-Heal)            │
├─────────────────────────────────────────────────────┤
│                  测试体系                            │
│  Unit Test (Mock LLM) → Integration Test → E2E     │
│  Eval Framework (Dataset + Scorer + Runner)         │
│  Regression Test (Baseline Comparison)              │
├─────────────────────────────────────────────────────┤
│                  文档体系                            │
│  OpenAPI Spec → Architecture Doc → API Doc           │
│  Ops Manual → Version Control → Quality Check       │
└─────────────────────────────────────────────────────┘
```

---

## 十、下一步

W11 完成，进入 **W12：企业级实战**（最终周）。

W12 将把 W1-W11 所学全部整合到一个企业级项目中，包括：
- 需求分析与架构设计
- 多 Agent 系统开发
- 测试与评测
- CI/CD 与部署
- 监控与运维
- 文档与交付