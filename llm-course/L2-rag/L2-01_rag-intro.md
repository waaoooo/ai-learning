# L2-01 · RAG 前言：LLM 缺陷 / RAG 定义 / 三大范式

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：理解为什么 LLM 单独不够用，RAG 是什么、解决什么、有哪几类范式；跑通一个最小 RAG。
> **先修**：已完成 Week1（LLM 调用 / Prompt / 多轮对话）；了解 embedding 向量概念（本课会复述）。
> **预计时长**：45-60 分钟

---

## 1. LLM 的缺陷（为什么需要 RAG）

大语言模型（LLM）在预训练阶段"吞下"了海量文本，但有几个硬伤，直接决定了它不能单独扛起企业应用：

| 缺陷 | 表现 | 后果 |
|---|---|---|
| **知识截止** | 训练数据有截止日期 | 问"昨天的新闻/最新政策"答不出或瞎编 |
| **私有数据盲区** | 训练集不包含你公司文档 | 问"我们公司的报销流程"答非所问 |
| **幻觉（Hallucination）** | 流畅但编造事实 | 企业场景不可接受 |
| **上下文窗口有限** | 单次输入有 token 上限 | 无法把整本手册塞进去 |
| **不可追溯** | 不知道答案来自哪 | 无法审计、无法更新 |
| **更新成本高** | 微调/重训昂贵 | 知识一变就要重训 |

**关键洞察**：LLM 擅长"理解与生成"，不擅长"记忆与更新"。把"记忆"外置给一个可检索的知识库，就是 RAG 的动机。

---

## 2. RAG 是什么

**RAG = Retrieval-Augmented Generation（检索增强生成）**。

核心思想：在 LLM 生成答案之前，先从外部知识库**检索**与问题相关的片段，把它们作为上下文喂给 LLM，让 LLM 基于这些证据生成答案。

```
用户问题
   │
   ▼
┌──────────┐   向量检索    ┌──────────────┐
│  Query   │ ───────────▶ │  Knowledge   │
│          │ ◀─────────── │  Base (向量库) │
└──────────┘   top-k 片段  └──────────────┘
   │
   ▼
┌──────────────────────────────┐
│ Prompt = 问题 + 检索到的片段  │
│          ↓                   │
│         LLM 生成答案           │
└──────────────────────────────┘
   │
   ▼
带引用的答案
```

**RAG 带来的好处**：
- 知识可更新（改文档即可，不重训模型）
- 答案可追溯（附来源）
- 降低幻觉（证据约束）
- 私有数据可用（企业文档、内部 wiki）
- 成本可控（比微调便宜）

---

## 3. RAG 三大范式

业界常见把 RAG 演进分为三类（本课采用此分类；不同资料命名略有差异）：

### 3.1 Naive RAG（朴素 RAG）
- 流程：切分文档 → 向量化 → top-k 检索 → 拼接 prompt → 生成
- 优点：简单、易实现
- 缺点：切分粒度粗糙、检索噪声大、上下文冗余、幻觉仍存

### 3.2 Advanced RAG（高级 RAG）
在 Naive 基础上做**检索前/中/后**优化：
- **Pre-Retrieval（预检索）**：查询改写、多路召回、索引优化（摘要/父子/元数据索引）
- **Retrieval（检索）**：混合检索（向量 + 关键词 BM25）
- **Post-Retrieval（后检索）**：重排序（Reranker）、上下文压缩、RAG-Fusion

### 3.3 Modular RAG（模块化 RAG）
把 RAG 拆成可插拔模块（路由、调度、融合、验证等），按需组合，支持多轮/Agent 式检索。可视为 Advanced RAG 的工程化形态。

> 📌 本 L2 课程按"Naive → 痛点 → Advanced → 评估 → 开源项目 → LangChain 实战"递进。

---

## 4. 最小可运行示例：Hello RAG

用 GLM（OpenAI 兼容）+ faiss + numpy，跑一个"基于内存小语料的问答"。

**前置**：`week1/.env` 中配置 `ZHIPU_API_KEY`。

配套脚本：`L2-01_hello_rag.py`（同目录）。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-01_hello_rag.py
```

核心代码（节选）：

```python
import os, numpy as np, faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../../week1/.env")
client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4")

# 1. 语料
docs = [
    "公司年假：入职满1年享有5天年假，满3年10天，满5年15天。",
    "报销流程：员工提交发票→直属经理审批→财务复核→3个工作日到账。",
    "加班补偿：工作日加班按1.5倍，周末2倍，法定节假日3倍。",
]

# 2. Embedding
def embed(texts):
    r = client.embeddings.create(model="embedding-3", input=texts)
    return np.array([d.embedding for d in r.data], dtype="float32")

V = embed(docs)
index = faiss.IndexFlatIP(V.shape[1])
faiss.normalize_L2(V)
index.add(V)

# 3. 检索
q = "年假怎么算？"
qv = embed([q]); faiss.normalize_L2(qv)
D, I = index.search(qv, 2)
ctx = "\n".join(docs[i] for i in I[0])

# 4. 生成
resp = client.chat.completions.create(
    model="glm-4-flash",
    messages=[
        {"role": "system", "content": "基于提供的资料回答，资料不足请说不知道。"},
        {"role": "user", "content": f"资料：\n{ctx}\n\n问题：{q}"},
    ],
)
print(resp.choices[0].message.content)
```

**运行结果预期**：模型回答年假天数规则，并基于检索到的片段，而非凭空编造。

> ⚠️ 若智谱 embedding 账户无余额（429），脚本会自动降级为本地字符 n-gram 伪向量演示，检索仍可跑通；生产请充值或换用其他 embedding API。Chat 生成不受影响。

---

## 5. 小结

- LLM 的弱项是"记忆/更新/追溯"，RAG 把记忆外置为可检索知识库。
- RAG = 检索 top-k 片段 → 拼入 prompt → LLM 生成，答案可追溯。
- 三范式：Naive → Advanced → Modular，本课按此递进。
- 最小 RAG 仅需：embedding + 向量库 + top-k + prompt 拼接。

---

## 6. 自测题

1. RAG 解决 LLM 的哪三类核心缺陷？（知识截止 / 私有数据 / 幻觉……）
2. Naive RAG 与 Advanced RAG 的关键差异是什么？
3. 为什么 top-k 检索后还要"重排序（rerank）"？（提示：向量相似度 ≠ 语义相关性）
4. 在示例中，如果把 `k=2` 改成 `k=5`，对答案质量有什么影响？

---

## 7. 延伸阅读

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020) — RAG 原论文
- 智谱 Embedding 文档：https://open.bigmodel.cn/dev/api#vector
- faiss 入门：https://github.com/facebookresearch/faiss/wiki

---

## 8. 验收动作

- [ ] 跑通 `L2-01_hello_rag.py`，输出正确回答年假规则
- [ ] 修改语料为公司真实文档（或任意 5 条），验证检索+生成链路
- [ ] 把 `k` 改为 1 和 5，观察答案差异，记录在笔记

---

## 附：配套脚本

见同目录 `L2-01_hello_rag.py`。
