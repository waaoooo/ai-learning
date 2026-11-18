# L2-08 · Advanced RAG：Pre-Retrieval（索引优化 / 查询优化）

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：掌握 Advanced RAG 的"检索前"优化——索引侧（摘要/父子/假设性问题/元数据）与查询侧（Enrich / Multi-Query / 问题分解 / HyDE），并能组合使用。
> **先修**：L2-01 ~ L2-07（Naive RAG + 痛点）。
> **预计时长**：90-120 分钟（含实验）

---

## 1. Advanced RAG 原理

Naive RAG 只做"切分→检索→生成"。Advanced RAG 把检索拆成三段：

```
Pre-Retrieval（检索前）→ Retrieval（检索）→ Post-Retrieval（检索后）
```

| 阶段 | 优化对象 | 代表技术 |
|---|---|---|
| Pre-Retrieval | 库 + 查询 | 父子索引、Multi-Query、HyDE |
| Retrieval | 检索本身 | 混合检索、ANN 参数 |
| Post-Retrieval | 检索结果 | rerank、RAG-Fusion、压缩 |

本课讲 Pre-Retrieval：**在查询到达前，优化"库"和"查询"本身**。

---

## 2. 索引优化（Indexing）

| 技术 | 做法 | 解决的问题 | 成本 |
|---|---|---|---|
| **摘要索引** | 对每个 chunk 生成摘要，用摘要检索，返回原文 | 长文检索噪声大 | 中 |
| **父子索引** | 小块（child）用于检索，命中后返回父块（parent） | 检索粒度细但上下文完整 | 中 |
| **假设性问题索引** | 为每个 chunk 生成"它可能回答的问题"，用问题做索引 | 查询与文档表述差异 | 中 |
| **元数据索引** | 标题、标签、日期等作为可过滤字段 | 精准过滤/路由 | 低 |

### 2.1 父子索引（最常用）
- **child 小（200-400 token）**：检索精度高
- **parent 大（800-1200 token）**：生成时上下文完整
- **流程**：检索 child → 映射到 parent → 把 parent 交给 LLM

### 2.2 摘要索引
- 对长 chunk 生成摘要，用摘要 embedding 检索
- 适合"文档很长但答案集中"的场景

### 2.3 假设性问题索引
- 为每个 chunk 生成若干"可能的问题"，用问题 embedding 建索引
- 用户问题直接命中"假设问题"，弥合表述差异

---

## 3. 查询优化（Query）

| 技术 | 做法 | 解决的问题 |
|---|---|---|
| **Enrich（查询增强）** | 用 LLM 补全/改写问题，补上下文 | 问题含糊、指代不清 |
| **Multi-Query（多路召回）** | 一个问题生成 3-5 个变体，分别检索后合并 | 单一查询表述偏差 |
| **Decomposition（问题分解）** | 复杂问题拆成子问题，分别检索再综合 | 多跳/复杂问题 |
| **HyDE** | 生成"假设答案文档"，用其 embedding 检索 | 查询-文档语义 gap |

### 3.1 示例：问"我们公司的年假和病假有什么区别？"
- **Enrich** → "年假（带薪年假）与病假在申请条件、天数、待遇上的区别"
- **Multi-Query** → ["年假和病假区别", "病假申请条件", "年假天数规定"]
- **Decomposition** → ["年假规则", "病假规则", "两者差异"]

### 3.2 实现（Multi-Query）
```python
def multi_query(question, llm, n=3):
    variants = llm(f"生成关于'{question}'的{n}个不同表述，每行一个")
    return [question] + variants.strip().split("\n")

results = []
for q in multi_query(question, llm):
    results += vector_search(q, k=5)
merged = dedup_by_doc(results)   # 按文档去重合并
```

### 3.3 HyDE 流程
```
问题 → LLM 生成假设答案 → 对假设答案做 embedding → 检索 → 返回真实文档
```
- 直觉：假设答案与真实文档在语义空间更接近

---

## 4. 技术选型与组合

| 问题类型 | 推荐技术 |
|---|---|
| 长文档、上下文要完整 | 父子索引 |
| 表述差异大 | 假设性问题索引 / HyDE / Multi-Query |
| 问题含糊 | Enrich |
| 多跳/复合 | Decomposition |

> 可叠加：父子索引 + Multi-Query 是常见高性价比组合。

---

## 5. 最小可运行示例

配套脚本：`L2-08_pre_retrieval.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-08_pre_retrieval.py
```

脚本演示：父子索引（child 检索 → 返回 parent）、Multi-Query（变体检索 + 合并去重）。

---

## 6. 实验与练习

1. 用 L2-05 的 HR 语料加父子索引，对比检索效果。
2. 对 3 个复杂问题做 Decomposition，记录子问题与检索结果。
3. 实现 HyDE，对比与直接检索的召回差异。
4. 把 Multi-Query 的变体数从 3 调到 5，观察召回与成本。

---

## 7. 小结

- Pre-Retrieval 优化"库"和"查询"两侧；父子索引是性价比最高的索引优化。
- Multi-Query / Decomposition 显著提升复杂问题召回；Enrich 处理含糊查询。
- 索引与查询优化可叠加，按需组合。

---

## 8. 自测题

1. 父子索引中，child 和 parent 的大小关系？为什么？
2. 假设性问题索引与 Multi-Query 的异同？
3. 什么问题适合 Decomposition？举例。
4. Enrich 与 HyDE 的区别？

### 自测题答案

1. **大小关系**：child 小（200-400 token）用于检索，保证精度；parent 大（800-1200 token）用于生成，保证上下文完整。
2. **异同**：同——都让"问题"参与检索，弥合查询-文档表述差异；异——假设性问题索引在索引侧为每个 chunk 生成可能的问题；Multi-Query 在查询侧生成多个变体。
3. **适合 Decomposition**：多跳/复合问题，如"A 公司 CEO 毕业于哪所大学？"（先查 CEO 再查学历）；"对比 X 和 Y 的优缺点"（拆成 X、Y、对比）。
4. **Enrich vs HyDE**：Enrich 直接补全/改写查询文本；HyDE 先让 LLM 生成"假设答案文档"，用其 embedding 检索，再返回真实文档。

---

## 9. 延伸阅读

- LlamaIndex "Sub Question Query Engine" 文档
- LangChain "Multi-query retriever" 文档
- "HyDE: Precise Zero-Shot Dense Retrieval" (Gao et al., 2022)

---

## 10. 验收动作

- [ ] 跑通 `L2-08_pre_retrieval.py`，观察父子索引与 multi-query 输出
- [ ] 用 L2-05 的 HR 语料加父子索引，对比检索效果
- [ ] 对 3 个复杂问题做 Decomposition，记录子问题

---

## 附录：FAQ

**Q：父子索引的 child/parent 如何映射？**
A：建库时记录每个 child 的 parent id；检索 child 后按映射取 parent。

**Q：Multi-Query 会不会很贵？**
A：检索调用翻倍（每变体一次），但向量检索便宜；生成仍一次。可通过缓存变体控制。

**Q：HyDE 的假设答案不准会怎样？**
A：可能引入偏差；可结合 Multi-Query/真实查询一起检索，取并集。

**Q：Pre-Retrieval 和 Post-Retrieval 都要做吗？**
A：按需。先诊断（L2-07/L2-11）定位瓶颈，再决定在哪一段加优化。
