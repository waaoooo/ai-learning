# L2-08 · Advanced RAG：Pre-Retrieval（索引优化 / 查询优化）

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：掌握 Advanced RAG 的"检索前"优化——索引侧（摘要/父子/假设性问题/元数据）与查询侧（Enrich / Multi-Query / 问题分解）。
> **先修**：L2-01 ~ L2-07（Naive RAG + 痛点）。
> **预计时长**：60-90 分钟

---

## 1. Advanced RAG 原理

Naive RAG 只做"切分→检索→生成"。Advanced RAG 把检索拆成三段：

```
Pre-Retrieval（检索前）→ Retrieval（检索）→ Post-Retrieval（检索后）
```

本课讲 Pre-Retrieval：在查询到达前，优化"库"和"查询"本身。

---

## 2. 索引优化（Indexing）

| 技术 | 做法 | 解决的问题 |
|---|---|---|
| **摘要索引** | 对每个 chunk 生成摘要，用摘要检索，返回原文 | 长文检索噪声大 |
| **父子索引** | 小块（child）用于检索，命中后返回其父块（parent） | 检索粒度细但上下文完整 |
| **假设性问题索引** | 为每个 chunk 生成"它可能回答的问题"，用问题做索引 | 查询与文档表述差异 |
| **元数据索引** | 标题、标签、日期等作为可过滤字段 | 精准过滤/路由 |

**父子索引**是最常用的：child 小（200-400 token）保证检索精度，parent 大（800-1200 token）保证生成时上下文完整。

---

## 3. 查询优化（Query）

| 技术 | 做法 | 解决的问题 |
|---|---|---|
| **Enrich（查询增强）** | 用 LLM 补全/改写问题，补上下文 | 问题含糊、指代不清 |
| **Multi-Query（多路召回）** | 一个问题生成 3-5 个变体，分别检索后合并 | 单一查询表述偏差 |
| **Decomposition（问题分解）** | 复杂问题拆成子问题，分别检索再综合 | 多跳/复杂问题 |

**示例**：问"我们公司的年假和病假有什么区别？"
- Enrich → "年假（带薪年假）与病假在申请条件、天数、待遇上的区别"
- Multi-Query → ["年假和病假区别", "病假申请条件", "年假天数规定"]
- Decomposition → ["年假规则", "病假规则", "两者差异"]

---

## 4. 最小可运行示例

配套脚本：`L2-08_pre_retrieval.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-08_pre_retrieval.py
```

脚本演示：
1. **父子索引**：child 检索 → 返回 parent
2. **Multi-Query**：一个查询生成 3 个变体，分别检索后合并去重

---

## 5. 小结

- Pre-Retrieval 优化"库"和"查询"两侧；父子索引是性价比最高的索引优化。
- Multi-Query / Decomposition 显著提升复杂问题召回；Enrich 处理含糊查询。
- 索引与查询优化可叠加，按需组合。

---

## 6. 自测题

1. 父子索引中，child 和 parent 的大小关系？为什么？
2. 假设性问题索引与 Multi-Query 的异同？
3. 什么问题适合 Decomposition？举例。
4. Enrich 与 HyDE（假设文档嵌入）的区别？

---

## 7. 延伸阅读

- LlamaIndex "Sub Question Query Engine" 文档
- LangChain "Multi-query retriever" 文档
- "HyDE: Precise Zero-Shot Dense Retrieval" (Gao et al., 2022)

---

## 8. 验收动作

- [ ] 跑通 `L2-08_pre_retrieval.py`，观察父子索引与 multi-query 输出
- [ ] 用 L2-05 的 HR 语料加父子索引，对比检索效果
- [ ] 对 3 个复杂问题做 Decomposition，记录子问题
