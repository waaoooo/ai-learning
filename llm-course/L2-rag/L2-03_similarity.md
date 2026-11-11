# L2-03 · Embedding 向量化与相似度（cosine / L2）+ top-k

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：理解文本如何变成向量（embedding）、如何度量向量相似度（cosine / L2 / 内积），以及 top-k 检索的实现。
> **先修**：L2-01（RAG 流程）、L2-02（分块）。了解向量基本概念。
> **预计时长**：45-60 分钟

---

## 1. 什么是 Embedding

Embedding 是把文本映射到高维向量空间的函数：语义相近的文本，向量距离近。

```
"年假怎么算？"  →  [0.12, -0.05, 0.88, ..., 0.03]   (dim=1024/1536/...)
"公司年假天数"  →  [0.11, -0.04, 0.85, ..., 0.02]   (距离近)
"报销流程"      →  [-0.30, 0.70, 0.10, ..., -0.20]   (距离远)
```

**关键性质**：
- 维度通常 768-3072（模型决定）
- 向量已归一化（模长=1）时，内积 = cosine 相似度
- Embedding 模型 ≠ LLM：专用小模型（bge、text-embedding-3），更快更便宜

---

## 2. 相似度度量

设向量 **a**, **b**：

- **余弦相似度 (Cosine)**：`cos(a,b) = (a·b) / (||a|| ||b||)`，范围 [-1,1]，值越大越相似。**文本检索首选**，对向量模长不敏感。
- **欧氏距离 (L2)**：`||a-b||`，值越小越相似。对模长敏感。
- **内积 (Dot Product)**：`a·b`。向量已归一化时 ≡ cosine；未归一化时偏向模长大的向量。

**选型建议**：
- 文本语义检索 → cosine（或归一化后的内积，faiss IndexFlatIP）
- 图像/多模态 → 常用 cosine
- 需要考虑"强度/重要性" → 内积

---

## 3. top-k 检索

给定查询向量 **q**，在向量库中找与 **q** 最相似的 k 个向量：

```
for each doc vector v_i:
    score_i = sim(q, v_i)
sort desc → take top-k
```

- k 通常 3-10；太大引入噪声，太小漏召回
- 实践：先 top-k（如 20）粗召回，再 rerank 精排（L2-09）

---

## 4. 最小可运行示例

用本地 fallback embedding（字符 n-gram）演示 cosine / L2 计算与 top-k；若 embedding API 可用则自动用真实 embedding。

配套脚本：`L2-03_similarity.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-03_similarity.py
```

---

## 5. 小结

- Embedding 把文本映射到高维向量；语义相近 → 向量近。
- cosine 是文本检索的默认度量；归一化后内积 = cosine，可用 faiss IndexFlatIP 加速。
- top-k 是检索的基本操作；k 的选择影响召回/噪声权衡。

---

## 6. 自测题

1. cosine 与 L2 在什么条件下排序结果一致？
2. 为什么生产中常用"归一化 + 内积"而不是直接 cosine？
3. k 设为 1 vs 10，对 RAG 答案质量的影响？
4. Embedding 模型与 LLM 的区别是什么？能否用 LLM 做 embedding？

自测题答案
- cosine 与 L2 何时排序一致？ — 向量已归一化（模长=1）时，L2 距离排序与 cosine 相似度排序单调等价（cosine 越大 ⇔ L2 越小）；对等模长向量两者等价。
- 为什么生产常用"归一化 + 内积"？ — 内积计算更快（省去除法/模长），faiss IndexFlatIP 原生支持；归一化后内积 ≡ cosine，精度等价。
- k=1 vs k=10 对答案质量？ — k=1 召回不足、易漏信息；k=10 引入噪声、稀释上下文、增 token 成本与幻觉风险；需按场景调优，常配合 rerank。
- Embedding 模型 vs LLM？能否用 LLM 做 embedding？ — Embedding 模型专用、小、快、便宜，输出固定维度向量；LLM 大、生成文本。可用 LLM 隐层做 embedding（last-token pooling），但成本高、效果不一定更好，生产多用专用 embedding 模型。

---

## 7. 延伸阅读

- OpenAI Embeddings guide: https://platform.openai.com/docs/guides/embeddings
- 智谱 embedding-3 文档
- faiss IndexFlatIP / IndexFlatL2: https://github.com/facebookresearch/faiss/wiki

---

## 8. 验收动作

- [ ] 跑通 `L2-03_similarity.py`，输出 cosine/L2/top-k 结果
- [ ] 自己加 3 条语料，验证检索排序合理性
- [ ] 将 embedding 换为真实 API（充值后），对比排序差异
