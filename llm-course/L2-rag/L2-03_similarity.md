# L2-03 · Embedding 向量化与相似度（cosine / L2）+ top-k

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：理解文本如何变成向量（embedding）、如何度量向量相似度（cosine / L2 / 内积），以及 top-k 检索的实现与调参。
> **先修**：L2-01（RAG 流程）、L2-02（分块）。
> **预计时长**：90-120 分钟（含实验）

---

## 1. 什么是 Embedding

Embedding 是把文本映射到高维向量空间的函数：**语义相近的文本，向量距离近**。

```
"年假怎么算？"  →  [0.12, -0.05, 0.88, ..., 0.03]   (dim=1024/1536/...)
"公司年假天数"  →  [0.11, -0.04, 0.85, ..., 0.02]   (距离近)
"报销流程"      →  [-0.30, 0.70, 0.10, ..., -0.20]   (距离远)
```

**关键性质**：
- 维度通常 768-3072（模型决定）
- 向量已归一化（模长=1）时，内积 = cosine 相似度
- Embedding 模型 ≠ LLM：专用小模型（bge、text-embedding-3），更快更便宜
- 语义空间是有"方向"的：`king - man + woman ≈ queen`（经典类比）

---

## 2. 相似度度量

设向量 **a**, **b**：

### 2.1 余弦相似度 (Cosine)
```
cos(a,b) = (a·b) / (||a|| ||b||)
```
- 范围 [-1,1]，值越大越相似
- **对模长不敏感**，只看方向 → 文本检索首选

### 2.2 欧氏距离 (L2)
```
L2(a,b) = ||a - b||
```
- 值越小越相似
- 对模长敏感；向量归一化后，L2 与 cosine 排序单调等价

### 2.3 内积 (Dot Product)
```
dot(a,b) = a·b
```
- 向量已归一化时 ≡ cosine
- 未归一化时偏向模长大的向量

### 2.4 三者关系
| 条件 | 结论 |
|---|---|
| 向量归一化 | cosine 与内积等价；L2 排序与 cosine 单调等价 |
| 向量未归一化 | 三者可能给出不同排序，内积偏向大模长 |

### 2.5 选型建议
- 文本语义检索 → **cosine**（或归一化后的内积，faiss `IndexFlatIP`）
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
- 实践：**先 top-k（如 20）粗召回，再 rerank 精排**（L2-09）
- k 的影响：
  - k=1：召回不足、易漏信息
  - k=10：引入噪声、稀释上下文、增 token 成本与幻觉风险
  - 需按场景调优，常配合 rerank

---

## 4. 向量维度与模型选择

| 模型 | 维度 | 特点 |
|---|---|---|
| bge-large-zh | 1024 | 中文强、可本地 |
| text-embedding-3-small/large | 1536/3072 | OpenAI、便宜/强 |
| embedding-3（智谱） | 2048 | 中文可用 |
| m3e | 768 | 轻量中文 |

- 维度越高 → 表达越强，但存储/计算成本越高
- 选型看：语言、成本、是否需本地部署

---

## 4.5 工程细节与常见坑

### 4.5.1 向量归一化
```python
import numpy as np
def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
```
- 归一化后内积即 cosine，可用整数/浮点索引加速
- 注意零向量（空文本）除零保护

### 4.5.2 相似度计算的数值
- 浮点误差：cosine 可能略超 [-1,1]，比较时用 `np.clip`
- 大规模检索：用矩阵乘法 `V @ q` 一次性算所有分数，避免循环
- 内存：100 万 × 1536 维 float32 ≈ 6GB，需量化或分片

### 4.5.3 Embedding API 调用示例
```python
from openai import OpenAI
client = OpenAI(api_key=KEY, base_url=BASE)
r = client.embeddings.create(model="embedding-3", input=["年假怎么算？"])
vec = r.data[0].embedding
```

### 4.5.4 常见坑
| 坑 | 表现 | 修法 |
|---|---|---|
| 查询与文档长度差异大 | 长文档模长大，内积偏向它 | 归一化用 cosine |
| 空文本 | 零向量/报错 | 过滤或占位向量 |
| 维度不匹配 | 检索报错 | 统一模型/统一维度 |
| 未归一化建索引 | 分数不可比 | 建索引前归一化 |

---

## 5. 最小可运行示例

用本地 fallback embedding（字符 n-gram）演示 cosine / L2 计算与 top-k；若 embedding API 可用则自动用真实 embedding。

配套脚本：`L2-03_similarity.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-03_similarity.py
```

---

## 6. 实验与练习

1. 自己加 3 条语料，验证检索排序合理性。
2. 把向量归一化后，验证 cosine 与 L2 排序一致。
3. 把 embedding 换为真实 API（充值后），对比排序差异。
4. 测试 k=1/3/10 的 top-k 结果差异。

---

## 7. 小结

- Embedding 把文本映射到高维向量；语义相近 → 向量近。
- cosine 是文本检索默认度量；归一化后内积 = cosine，可用 faiss `IndexFlatIP` 加速。
- top-k 是检索基本操作；k 的选择影响召回/噪声权衡。

---

## 8. 自测题

1. cosine 与 L2 在什么条件下排序结果一致？
2. 为什么生产中常用"归一化 + 内积"而不是直接 cosine？
3. k 设为 1 vs 10，对 RAG 答案质量的影响？
4. Embedding 模型与 LLM 的区别是什么？能否用 LLM 做 embedding？

### 自测题答案

1. **一致条件**：向量已归一化（模长=1）时，L2 距离排序与 cosine 相似度排序单调等价；对等模长向量两者等价。
2. **归一化 + 内积**：内积计算更快（省去除法/模长），faiss `IndexFlatIP` 原生支持；归一化后内积 ≡ cosine，精度等价。
3. **k=1 vs 10**：k=1 召回不足、易漏信息；k=10 引入噪声、稀释上下文、增 token 成本与幻觉风险；需按场景调优，常配合 rerank。
4. **Embedding vs LLM**：Embedding 模型专用、小、快、便宜，输出固定维度向量；LLM 大、生成文本。可用 LLM 隐层做 embedding（last-token pooling），但成本高、效果不一定更好，生产多用专用 embedding 模型。

---

## 9. 延伸阅读

- OpenAI Embeddings guide: https://platform.openai.com/docs/guides/embeddings
- 智谱 embedding-3 文档
- faiss IndexFlatIP / IndexFlatL2: https://github.com/facebookresearch/faiss/wiki

---

## 10. 验收动作

- [ ] 跑通 `L2-03_similarity.py`，输出 cosine/L2/top-k 结果
- [ ] 自己加 3 条语料，验证检索排序合理性
- [ ] 将 embedding 换为真实 API（充值后），对比排序差异

---

## 附录：FAQ

**Q：cosine 和 L2 该用哪个？**
A：文本检索默认 cosine（对模长不敏感）。向量已归一化时 L2 排序与 cosine 等价，可任选。

**Q：为什么不用 LLM 直接做 embedding？**
A：可用（取隐层向量），但慢、贵、维度大，效果未必优于专用 embedding 模型；生产用专用模型。

**Q：embedding 维度越高越好吗？**
A：不是。维度高表达强但存储/计算贵；按任务选（768-1536 常用）。

**Q：top-k 怎么定？**
A：从 3-10 起调，看 Recall@k 与噪声；生产常"粗召回 top-20 + rerank 精排"。

**Q：embedding 模型需要和 LLM 同厂商吗？**
A：不需要。embedding 与生成是独立环节，可混用（如 bge embedding + GLM 生成）。
