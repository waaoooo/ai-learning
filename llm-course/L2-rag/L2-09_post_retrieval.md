# L2-09 · Advanced RAG：混合检索 + Post-Retrieval

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：掌握检索中（混合检索）与检索后（重排序 / RAG-Fusion / 上下文压缩）的优化技术，理解各自原理、参数与工程权衡。
> **先修**：L2-01 ~ L2-08。
> **预计时长**：90-120 分钟（含实验）

---

## 1. 混合检索：本质 / 场景 / 实现

### 1.1 本质
**向量检索（语义）+ 关键词检索（精确）互补**。向量擅长"意思相近"，关键词擅长"字面精确"。

| 检索方式 | 强项 | 弱项 |
|---|---|---|
| 向量 | 同义改写、跨语言、语义相关 | 精确术语/缩写/编号/罕见词 |
| BM25 | 精确术语、编号、专有名词 | 语义改写、词面不同 |

### 1.2 适用场景
- 术语/缩写/编号多：**医疗、法律、代码、金融**
- 查询表述与文档差异大：口语化提问 vs 书面文档
- 需要高精度 top-k：先混合召回再 rerank

### 1.3 实现
```python
vec_hits = vector_search(q, k=20)      # 语义
kw_hits  = bm25_search(q, k=20)        # 关键词
final    = rrf_fuse(vec_hits, kw_hits) # 融合
```
融合方式见 L2-06（加权 / RRF / 两阶段）。

---

## 2. 重排序（Rerank）

粗召回（top-k）之后，用**更精确的模型**对少量候选重排。

### 2.1 Reranker 模型（cross-encoder）
- **原理**：把 `(query, doc)` 拼接后一起输入模型，输出相关性分数；query 与 doc 充分交互，精度高于双塔 embedding。
- **代价**：每个 `(q, d)` 都要过一次模型，O(n) 且慢 → **只能对少量候选（10-50 条）重排**。
- **代表**：Cohere Rerank、bge-reranker、智谱 rerank。

### 2.2 LongContextReorder（"lost in the middle"）
- **现象**：LLM 对长上下文的**中段**注意力衰减，相关 chunk 放中间会被忽略。
- **做法**：把最相关的 chunk 放在**开头和结尾**。
```
原始顺序: [c1, c2, c3, c4, c5]   (c1/c5 相关)
重排后:   [c1, c3, c4, c5, c2]   → 相关项靠两端
```
- **适用**：上下文很长（>4 个 chunk）且相关性差异明显时。

### 2.3 其他重排
- **LLM rerank**：让 LLM 对候选打分/排序（listwise），效果最好但成本高
- **时间/权威性加权**：新文档、高权威来源加权

---

## 3. RAG-Fusion

### 3.1 Multi-Query
一个查询生成多个**变体**（不同表述/角度），分别检索，扩大召回覆盖。

### 3.2 RRF 融合
对每个变体的排名结果累加：
```
RRF_score(d) = Σ_i  1 / (k + rank_i(d))
```
- **k 的作用**：控制排名权重衰减。k 小（如 10）→ 头部差异被放大、偏向强匹配；k 大（如 60）→ 更平滑、融合更均衡。
- **优点**：只用 rank，无需归一化，天然融合不同尺度分数。

### 3.3 为什么有效
单一查询可能措辞不佳；多查询 + RRF 让"任一变体命中"的文档浮上来，召回更稳。

---

## 4. 上下文压缩与过滤

### 4.1 压缩
从检索到的长 chunk 中，只抽取**与问题相关的句子/片段**（抽取式摘要），减少 token、降低噪声。

```python
def compress(chunk, question, top=2):
    sents = chunk.split("。")
    scored = [(s, len(set(s) & set(question))) for s in sents]
    scored.sort(key=lambda x: -x[1])
    return "。".join(s for s, _ in scored[:top])
```

### 4.2 过滤
- 去重（相似 chunk 合并）
- 去低分（低于阈值）
- 去无关（与问题无关的段落）

### 4.3 风险与缓解
- **风险**：压缩丢关键信息、破坏语义完整性
- **缓解**：保留原句 + 压缩后校验 + 结合 rerank 分数 + 控制压缩率

---

## 5. 完整流水线

```
查询 → 混合检索(向量+BM25) → RRF 融合 → rerank(cross-encoder)
     → LongContextReorder → 上下文压缩/过滤 → LLM 生成（带引用）
```

---

## 6. 最小可运行示例

配套脚本：`L2-09_post_retrieval.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-09_post_retrieval.py
```

脚本演示：LongContextReorder、RAG-Fusion（多查询 + RRF）、上下文压缩。

---

## 7. 实验与练习

1. 对 L2-06 的混合检索结果加 rerank，对比 top-3 变化。
2. 实现句子级压缩函数，验证压缩率与答案质量。
3. 把 RRF 的 k 从 10 调到 60，观察排序变化。
4. 构造"相关 chunk 在中间"的场景，验证 LongContextReorder 的效果。

---

## 8. 小结

- 混合检索解决"语义 vs 精确"互补；Rerank 解决"粗召回噪声"。
- LongContextReorder 缓解 LLM 中段注意力衰减。
- RAG-Fusion + 上下文压缩提升上下文质量与成本效率。

---

## 9. 自测题

1. cross-encoder reranker 与双塔 embedding 的区别？为什么 reranker 不能用于全库检索？
2. LongContextReorder 解决什么问题？
3. RRF 的 k 参数作用？k 越大/越小会怎样？
4. 上下文压缩会带来什么风险？如何缓解？

### 自测题答案

1. **cross-encoder vs 双塔**：双塔分别编码 query/doc，可离线建库、检索快，用于粗召回；cross-encoder 将 query+doc 拼接输入、交互充分、精度高，但每个 (q,d) 都要过一次模型，O(n) 且慢，只能对少量候选重排。
2. **LongContextReorder**：解决 LLM 对长上下文中段注意力衰减（"lost in the middle"），把最相关 chunk 放到开头和结尾。
3. **RRF 的 k**：控制排名权重衰减。k 小（如 10）头部差异被放大、偏向强匹配；k 大（如 60）更平滑、融合更均衡。
4. **压缩风险**：可能丢关键信息、破坏语义完整性；缓解：保留原句、压缩后校验、结合 rerank 分数。

---

## 10. 延伸阅读

- "Lost in the Middle" (Liu et al., 2023)
- Cohere Rerank 文档
- LangChain "Contextual compression" 文档

---

## 11. 验收动作

- [ ] 跑通 `L2-09_post_retrieval.py`，观察三种技术输出
- [ ] 对 L2-06 混合检索结果加 rerank，对比 top-3
- [ ] 实现句子级压缩，验证压缩率与答案质量

---

## 附录：FAQ

**Q：reranker 选 cross-encoder 还是 LLM？**
A：cross-encoder 快、便宜、够用，生产首选；LLM rerank 效果上限高但慢贵，适合离线/高价值场景。

**Q：LongContextReorder 什么时候用？**
A：上下文较长（>4 chunk）且相关性差异明显时；短上下文收益小。

**Q：RAG-Fusion 会增加多少成本？**
A：Multi-Query 会让检索调用翻倍（每变体一次），但只在检索层，生成仍一次；可控。

**Q：压缩会丢答案吗？**
A：可能。务必保留原句、控制压缩率、压缩后校验，并保留 rerank 分数作兜底。

**Q：Post-Retrieval 三步都要上吗？**
A：按需。先诊断（L2-07/L2-11）再决定加哪一步，避免过度工程。
