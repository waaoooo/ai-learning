# Week 8 学习文档：可观测性与调试

> **学习日期**：2026-10-28 · **语言**：Python 3.9 · **模型**：智谱 GLM-4-Flash（免费）
> **文档性质**：过程性学习记录，可独立阅读
> **前置**：Week 7（评测体系深化）

---

## 一、本周目标与达成情况

**一句话目标**：让 Agent 系统从"黑盒"变成"白盒" —— 出问题时能**定位到具体哪一步、哪个 Agent、哪次调用**。

| 任务 | 主题 | 状态 | 关键产出文件 |
|---|---|:---:|---|
| T1 | **为什么需要可观测性**（Agent 系统的黑盒问题） | ✅ | `01_why_observability.py` |
| T2 | **结构化日志**（JSON 日志、request_id 追踪） | ✅ | `02_structured_logging.py` |
| T3 | **链路追踪**（Trace/Span，多 Agent 调用链） | ✅ | `03_trace_span.py` |
| T4 | **性能分析**（token 计费、延迟分布、瓶颈定位） | ✅ | `04_performance.py` |
| T5 | 综合：可观测性仪表盘 | ✅ | `05_dashboard.py` |

**最终验收**：一个统一的可观测性系统，包含 6 个面板（Token/链路/准确率/性能/错误/资源）+ 告警自动触发。

---

## 二、本周最重要的认知

> ### 可观测性的本质是**把"猜"变成"读"**。
> 没有日志时，出错只能靠猜；有了日志，定位时间从"小时级"变成"秒级"。
>
> **最重要的一条**：**request_id 是链路追踪的基石** —— 没有它，日志就是散落的碎片；有了它，日志就是可追溯的证据链。

---

## 三、知识点详解

### 3.1 为什么需要可观测性（T1）

**黑盒问题**：多 Agent 系统中，一个请求经过 3 个 Agent + 汇总 + 格式化，出错了你只知道"最终答案错了"，不知道是哪一步。

**实测**：
```
黑盒模式：
  输入 "年假和病假各有多少天？" → 最终答案 "15"（错了，但你看不出来）
  是 step1 提取错了？step2 计算错了？还是 step3 格式化错了？

加日志后：
  req-001 Step1 extract: input="公司规定年假15天，病假30天。" → output="15\n30" (2266ms)
  req-001 Step2 calculate: input="15\n30" → output="ERROR: invalid input" (0ms)  ← 一眼定位！
  req-001 Step3 format: input="ERROR: invalid input" → output="抱歉，输入有误..." (1575ms)
```

**Bonus 发现**：Step 3 把 "12" 格式化成了一句哲学句子 —— 没有日志你根本不知道 Step 3 幻觉了。

### 3.2 结构化日志与 request_id（T2）

**文本日志 vs JSON 日志**：

| 维度 | 文本日志 | JSON 日志 |
|---|---|---|
| 可读性 | ✅ 人眼友好 | ⚠️ 需 jq 解析 |
| 可查询性 | ❌ grep + 正则 | ✅ `jq 'select(.status=="error")'` 一行搞定 |
| 可编程 | ❌ 难解析 | ✅ 直接导入 ELK/Grafana |
| 稳定性 | ❌ 格式变化就崩溃 | ✅ 固定字段，程序自动解析 |

**request_id 传递**：
```
req-A: Step1 → Step2 → Step3，全部带 req_id="req-A"
req-B: Step1 → Step2 → Step3，全部带 req_id="req-B"
```

**5 个设计要点**：
1. request_id 从入口生成，传给所有下游步骤
2. 固定字段：ts, req_id, step, name, input, output, ms, status
3. input/output 截断（避免日志过大）
4. JSON 格式 → 可被 jq/ELK/Grafana 直接消费
5. 可扩展字段（model, token_count, temperature）

**jq 查询示例**：
```bash
cat trace.jsonl | jq 'select(.status=="error")'                    # 找所有 ERROR
cat trace.jsonl | jq 'select(.req_id=="req-A" and .step==2)'       # 找 req-A 的 Step 2
cat trace.jsonl | jq 'select(.ms > 2000)'                           # 找耗时 > 2s 的
cat trace.jsonl | jq -s 'group_by(.name) | map({name: .[0].name, avg_ms: (map(.ms)|add/length)})'  # 平均耗时
```

### 3.3 链路追踪：Trace 与 Span（T3）

**数据模型**：
- **Trace** = 一次完整请求的端到端追踪（一个 trace_id）
- **Span** = Trace 内的一个工作单元（span_id + parent_span_id）

**Span 字段**：trace_id, span_id, parent_span_id, name, start_time, end_time, duration_ms, status, attributes

**调用树**（实测）：
```
✅ orchestrator (36480ms)
  ✅ agent.product (6372ms)
    ✅ llm.call (6372ms)
  ✅ agent.engineer (6337ms)
    ✅ llm.call (6337ms)
  ✅ agent.tester (6358ms)
    ✅ llm.call (6358ms)
  ✅ orchestrator.summarize (8397ms)
    ✅ llm.call (8397ms)
```

**关键发现**：orchestrator.summarize (8.4s) 比单个 Agent 还慢 —— 因为它把 3 个 Agent 的输出全塞进了 prompt。

**概念对比**：

| 概念 | 作用 | 类比 |
|---|---|---|
| request_id | 一条链路的唯一标识 | 订单号 |
| trace_id | 同 request_id，追踪整条链路 | 同上 |
| span | 链路中的一个工作单元 | 订单的每个步骤 |
| parent_span_id | 构建嵌套关系 | 步骤的父子关系 |

### 3.4 性能分析与瓶颈定位（T4）

**性能汇总**（实测）：

| 指标 | 值 |
|---|---|
| 总调用次数 | 5 |
| 总 Token | 954 (input=295, output=659) |
| 总成本 | 0.000162 元 |
| P50 延迟 | 7823ms |
| P95 延迟 | 9861ms |
| 延迟范围 | 2832ms ~ 9861ms |

**瓶颈定位**：

| 步骤 | 耗时 | Token | 成本 |
|---|---|---|---|
| agent.product | 9861ms | 178 | 0.000033 |
| agent.engineer | 9382ms | 177 | 0.000033 |
| agent.tester | 7823ms | 178 | 0.000033 |
| orchestrator.summarize | 7568ms | 293 | 0.000044 |
| format.output | 2832ms | 128 | 0.000019 |

**🔴 最慢**：agent.product (9861ms)
**💰 最贵**：orchestrator.summarize (0.000044 元) — 因为输入上下文最长

**优化建议**：

| 优化 | 预期效果 |
|---|---|
| 并行执行（asyncio） | 延迟 26s → 9.8s（3x） |
| 压缩 prompt | input tokens 减半，成本降 30% |
| 缓存 LLM 调用 | 重复请求直接返回，省 100% |
| 换模型 | 非关键步骤用 glm-4-flash（免费） |

**监控告警阈值**：

| 指标 | 阈值 | 动作 |
|---|---|---|
| P95 延迟 | > 10s | 告警 |
| 单次成本 | > 0.01 元 | 告警 |
| 错误率 | > 5% | 告警 |
| Token 消耗 | > 2000/次 | 告警 |

### 3.5 综合：可观测性仪表盘（T5）

**6 个面板**：

| 面板 | 内容 | 数据来源 |
|---|---|---|
| 1. Token 用量 | 总消耗 + 按步骤拆分 | PerformanceTracker |
| 2. 调用链路追踪 | Trace 树 + 耗时 | Span 列表 |
| 3. 准确率 | single vs multi 对比 | 评测结果 |
| 4. 性能监控 | P50/P95/Max/Min | 延迟分布 |
| 5. 错误/报警 | ERROR 日志 + 告警 | 告警规则 |
| 6. 资源使用 | CPU/MEM/GPU | 系统监控 |

**仪表盘实测**：
```
Token 用量: 总 820 (product=178, engineer=177, tester=178, summarize=287)
调用链路: orchestrator(28216ms) → 4 个子 Span
准确率: single=72%, multi=86%
性能: P50=4548ms, P95=5840ms
错误: agent.tester timeout → 告警触发 (error.rate=0.05 > 0.03)
资源: CPU=45.2%, MEM=1.2GB, GPU=0%
```

---

## 四、完整代码汇总

### 4.1 结构化日志器（T2）

```python
class TraceLogger:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.logs = []

    def log(self, step, name, input_text, output, ms, status="ok", **extra):
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "req_id": self.request_id,
            "step": step,
            "name": name,
            "input": input_text[:100],
            "output": output[:100],
            "ms": round(ms),
            "status": status,
        }
        entry.update(extra)
        self.logs.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
```

### 4.2 Span 追踪器（T3）

```python
class Span:
    def __init__(self, trace_id, name, parent_span_id=None):
        self.trace_id = trace_id
        self.span_id = uuid.uuid4().hex[:8]
        self.parent_span_id = parent_span_id
        self.name = name
        self.start_time = time.time()
        self.end_time = None
        self.status = "running"
        self.attributes = {}

    def end(self, status="ok"):
        self.end_time = time.time()
        self.status = status

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "duration_ms": round((self.end_time - self.start_time) * 1000),
            "status": self.status,
            "attributes": self.attributes,
        }
```

### 4.3 性能追踪器（T4）

```python
class PerformanceTracker:
    def __init__(self):
        self.records = []

    def record(self, step_name, prompt, response, input_tokens, output_tokens, latency_ms, model=MODEL):
        cost = (input_tokens * PRICING[model]["input"] +
                output_tokens * PRICING[model]["output"]) / 1000
        self.records.append({
            "step": step_name,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": round(latency_ms),
            "cost_yuan": round(cost, 6),
        })
```

### 4.4 统一可观测性系统（T5）

```python
class ObservabilitySystem:
    """统一的可观测性系统：日志 + 追踪 + 性能 + 告警"""

    def __init__(self):
        self.logs = []          # 结构化日志
        self.spans = []         # Span 列表
        self.metrics = []       # 性能指标
        self.alerts = []        # 告警记录

    def log(self, level, message, **fields): ...
    def start_span(self, name, trace_id, parent_id=None, **attrs): ...
    def end_span(self, span, status="ok", **attrs): ...
    def record_metric(self, name, value, unit="", tags=None): ...
    def check_alerts(self, thresholds): ...
    def dashboard(self): ...   # 6 面板一屏概览
```

---

## 五、学习过程记录（关键实验时间线）

| 阶段 | 做了什么 | 真实发现 |
|---|---|---|
| 1 | 黑盒 vs 白盒对比 | 没日志时出错只能猜；有日志一眼定位 Step 2 |
| 2 | 文本日志 vs JSON 日志 | JSON 可 jq 查询，文本日志无法高效查询 |
| 3 | request_id 传递 | 两条请求链路完全隔离，可独立追溯 |
| 4 | Trace/Span 调用树 | 9 个 Span，parent_span_id 构建树 |
| 5 | 性能追踪 | 总 954 tokens, 0.000162 元, P95 9861ms |
| 6 | 瓶颈定位 | agent.product 最慢 (9861ms), summarize 最贵 (0.000044 元) |
| 7 | 综合仪表盘 | 6 面板一屏概览，告警自动触发 |

---

## 六、踩坑记录（本周重点）

| 坑 | 现象 | 根因 | 修复 |
|---|---|---|---|
| **Step 3 幻觉** | 把 "12" 格式化成哲学句子 | LLM 对短输入过度发挥 | 加 system prompt 限制输出格式 |
| **input/output 截断** | 日志过大 | 未截断长文本 | input/output 各截断 100 字符 |
| **Span 嵌套错误** | 调用树显示不正确 | parent_span_id 传递错误 | 确保 start_span 时正确传入 parent |

> **本周教训**：
> 1. **日志不是越多越好** —— 截断 + 结构化比全量文本更有用。
> 2. **request_id 是链路追踪的基石** —— 没有它，日志就是散落的碎片。
> 3. **可观测性不是"事后排查"** —— 它是"事前预防"的工具（告警、监控）。

---

## 七、核心结论（记住这 6 条）

1. **可观测性的本质是把"猜"变成"读"** —— 没有日志时出错只能猜。
2. **结构化日志（JSON）优于文本日志** —— 可查询、可编程、可可视化。
3. **request_id 是链路追踪的基石** —— 没有它，日志就是散落的碎片。
4. **Trace = 一次完整请求，Span = 工作单元** —— parent_span_id 构建调用树。
5. **性能追踪：Token/延迟/成本** —— 找出最慢/最贵的步骤。
6. **告警自动触发** —— 基于阈值，事前预防。

> **一句话总结 W8**：从"黑盒"到"白盒"，从"出错后排查"到"事前预防"。

---

## 八、知识地图（W8 → W9 衔接）

```
W8 已掌握
├── 为什么需要可观测性（黑盒 → 白盒）
├── 结构化日志（JSON + request_id）
├── Trace/Span 链路追踪
├── 性能分析（Token/延迟/成本/瓶颈）
└── 综合仪表盘（6 面板 + 告警）

W9 要跨越：从"能排查"到"能部署"
└── 工程化部署
    ├── 本地模型量化（节省成本）
    ├── LangChain 部署（框架选型）
    └── CI/CD（自动化部署）
```

---

## 九、遗留问题（后续周次补）

| 遗留 | 补的周次 |
|---|---|
| **真并行执行**（当前是串行） | W10（架构设计与部署） |
| **成本控制**（多 Agent = token 翻倍） | W11（成本控制与优化） |
| **多 Agent 调试工具** | W8 已覆盖（可观测性） |
| **人在环路的产品化设计** | W10 |
| **生产环境接入 Grafana/Prometheus** | W10 |

---

## 附录：week8 文件清单

| 文件 | 用途 |
|---|---|
| `01_why_observability.py` | 黑盒 vs 白盒对比 |
| `02_structured_logging.py` | 结构化日志 + request_id |
| `03_trace_span.py` | Trace/Span 链路追踪 |
| `04_performance.py` | 性能分析与瓶颈定位 |
| `05_dashboard.py` | 综合可观测性仪表盘 |
