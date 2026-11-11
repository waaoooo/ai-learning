# L2-04 · 向量数据库选型与实战（faiss）

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：了解主流向量数据库的差异与选型原则，并用 faiss 完成索引构建、持久化、加载与检索的全流程。
> **先修**：L2-01 ~ L2-03（RAG 流程 / 分块 / embedding / 相似度）。
> **预计时长**：60 分钟

---

## 1. 为什么需要向量数据库

暴力检索（全库扫描）在数据量大时不可行；向量数据库提供：
- **ANN（近似最近邻）** 索引：HNSW、IVF、PQ，毫秒级检索百万/亿级向量
- **持久化与 CRUD**：增删改查、元数据过滤
- **多租户 / 分布式 / 高可用**（部分）

---

## 2. 主流向量数据库对比

| 数据库 | 类型 | 特点 | 适用 |
|---|---|---|---|
| **faiss** | 库（in-process） | Facebook 开源，纯向量检索，无服务端，最快 | 原型/单机/嵌入式 |
| **Chroma** | 嵌入式/服务 | 轻量、Python 友好、持久化简单 | 本地开发/小规模 |
| **Qdrant** | 服务 | Rust、过滤强、gRPC/REST、云托管 | 生产、需元数据过滤 |
| **Milvus** | 分布式服务 | 企业级、大规模、GPU 支持 | 大规模生产 |
| **Weaviate** | 服务 | GraphQL、多模态、模块化 | 多模态/语义搜索 |
| **pgvector** | PG 扩展 | 与业务库同库，SQL 联合查询 | 已用 Postgres 的团队 |
| **Pinecone** | 托管 SaaS | 免运维，按量计费 | 快速上线、无运维团队 |

---

## 3. 选型指南

- **原型/教学/单机** → faiss 或 Chroma
- **中小规模生产 + 需过滤** → Qdrant / pgvector
- **大规模/高并发** → Milvus
- **已有 Postgres** → pgvector（运维成本最低）
- **无运维团队** → Pinecone / Qdrant Cloud

**通用原则**：先用 faiss/Chroma 验证检索质量，再按规模与过滤需求升级。

---

## 4. faiss 索引类型

| 索引 | 精度 | 速度 | 内存 | 说明 |
|---|---|---|---|---|
| `IndexFlatL2` / `IndexFlatIP` | 精确 | 慢（O(n)） | 高 | 暴力检索，适合 <10万 |
| `IndexIVFFlat` | 近似 | 快 | 中 | 聚类分桶，需训练 |
| `IndexHNSWFlat` | 近似 | 很快 | 高 | 图索引，查询快、构建慢 |
| `IndexIVFPQ` | 近似 | 快 | 低 | 量化压缩，亿级 |

**实践**：数据 <10万用 Flat；10万-1000万用 HNSW/IVF；亿级用 IVFPQ + GPU。

---

## 5. 最小可运行示例

构建 faiss 索引 → 检索 → 保存到磁盘 → 重新加载 → 再检索。

配套脚本：`L2-04_faiss.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-04_faiss.py
```

---

## 6. 小结

- 向量数据库核心价值：ANN 加速 + 持久化 + 过滤。
- 选型按规模与运维能力；原型阶段 faiss 足够。
- faiss 提供 Flat/IVF/HNSW/PQ 等索引，按数据规模选择。

---

## 7. 自测题

1. faiss 与 Qdrant 的本质区别是什么？（库 vs 服务）
2. 数据量从 1 万增长到 1000 万，索引策略应如何变化？
3. 为什么 HNSW 查询快但构建慢？
4. 元数据过滤（如"只检索 2024 年文档"）在 faiss 中如何实现？

自测题答案
- faiss 与 Qdrant 的本质区别？ — faiss 是 in-process 库（无服务端、无持久化/过滤/多租户）；Qdrant 是独立服务（REST/gRPC、元数据过滤、持久化、高可用）。
- 数据量 1 万 → 1000 万，索引策略如何变化？ — 1 万用 Flat（精确）；10 万+ 用 HNSW/IVF（近似）；1000 万+ 用 IVFPQ + 量化/GPU 降内存。
- HNSW 为什么查询快但构建慢？ — 图结构导航，查询 O(log n) 跳跃；构建需为每个节点建立多层邻居连接，O(n log n)，内存开销大。
- 元数据过滤在 faiss 中如何实现？ — faiss 本身不支持；需先按元数据过滤 doc id 列表再在子集检索，或换用支持过滤的库（Qdrant/Milvus/pgvector）。

---

## 8. 延伸阅读

- faiss wiki: https://github.com/facebookresearch/faiss/wiki
- Qdrant docs: https://qdrant.tech/documentation/
- pgvector: https://github.com/pgvector/pgvector

---

## 9. 验收动作

- [ ] 跑通 `L2-04_faiss.py`，完成构建/检索/保存/加载全流程
- [ ] 尝试 IndexHNSWFlat，对比与 Flat 的速度差异
- [ ] 用 L2-02 的 chunk 结果构建索引，端到端检索
