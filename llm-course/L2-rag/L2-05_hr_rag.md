# L2-05 · 应用实战：HR 制度智能问答（向量检索）

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：把 L2-01~L2-04 串起来，实现一个端到端的 HR 制度问答系统：文档 → 分块 → embedding → 索引 → 检索 → 生成（带引用），并掌握生产级工程点。
> **先修**：L2-01 ~ L2-04。
> **预计时长**：90-120 分钟（含实验）

---

## 1. 系统架构

```
HR 制度文档（md/txt）
   ↓  加载 + 分块 (L2-02)
chunks[] + metadata（制度名/条目号）
   ↓  embedding (L2-03)
vectors
   ↓  faiss IndexFlatIP (L2-04)
index
   ↓  top-k 检索（可加阈值）
相关 chunks + 来源
   ↓  prompt 拼接（仅基于资料 + 引用）
LLM 生成答案（带引用）
   ↓  无命中降级
"未找到相关制度"
```

---

## 2. 关键工程点

### 2.1 分块与 metadata
- 按制度条目切分，保留"制度名称"作为 metadata
- 好处：引用来源、按制度过滤、更新时精确定位

### 2.2 检索
- `top-k=3~5`；加**相似度阈值**过滤低相关结果
- 阈值要按 embedding 分布调（cosine 0.5-0.7 常见起点）

### 2.3 Prompt 设计
```python
prompt = f"""你是HR助手。仅根据以下制度资料回答，不得编造。
若资料不足，请回答"未找到相关制度"。
回答末尾注明来源（制度名）。

资料：
{context}

问题：{question}
"""
```

### 2.4 引用
- 答案末尾列出来源 chunk id / 制度名
- 价值：可追溯、可审计、降低幻觉信任、便于更新定位

### 2.5 降级
- 无相关结果 → 返回"未找到"，避免把无关上下文喂给 LLM 产生幻觉
- 可加 fallback：转人工 / 引导换个问法

---

## 3. 生产化清单

| 环节 | 生产要求 |
|---|---|
| 索引 | 持久化（faiss 存盘 / 换向量库） |
| 更新 | 制度变更后增量更新 chunk，不重建全库 |
| 权限 | 按部门/角色过滤可检索制度 |
| 审计 | 记录 query / 命中 / 答案 / 引用 |
| 成本 | 控制 top-k 与上下文长度 |
| 评估 | 建评测集回归（L2-11） |

---

## 4. 最小可运行示例

配套脚本：`L2-05_hr_rag.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-05_hr_rag.py
```

脚本内置 8 条 HR 制度，对 4 个测试问题跑问答，打印检索片段与答案；并含"无答案"降级演示。

---

## 4.5 完整参考实现（骨架）

```python
import numpy as np

def build_index(docs):
    chunks = []
    for name, text in docs.items():
        for i in range(0, len(text), 400):
            chunks.append({"text": text[i:i+400], "source": name})
    vecs = embed([c["text"] for c in chunks])       # 真实 embedding
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs.astype("float32"))
    return index, chunks

def answer(question, index, chunks, llm, k=4, threshold=0.5):
    qv = embed([question]).astype("float32")
    scores, ids = index.search(qv, k)
    hits = [chunks[i] for s, i in zip(scores[0], ids[0]) if s >= threshold]
    if not hits:
        return "未找到相关制度。"
    context = "\n".join(h["text"] for h in hits)
    ans = llm(prompt(question, context))
    sources = "、".join(sorted({h["source"] for h in hits}))
    return f"{ans}\n\n来源：{sources}"
```

要点：**阈值过滤 + 无命中降级 + 引用来源**是三个必做项。

---

## 5. 实验与练习

1. 新增 3 条制度 + 2 个问题，验证泛化。
2. 故意问库中没有的问题，验证"不知道"降级。
3. 调整相似度阈值，观察答案变化。
4. 把 top-k 从 3 调到 5，观察引用质量。

---

## 6. 小结

- 端到端 RAG = 加载 → 分块 → embedding → 索引 → 检索 → 生成；每步都是 L2-01~04 的组合。
- 引用与"不知道"降级是生产 RAG 的基本素养。
- 本例用内存语料 + faiss，生产需持久化索引与文档库。

---

## 7. 自测题

1. 为什么 HR 问答要"附引用"？对用户和审计的价值？
2. 如果检索到的 top-3 都与问题无关，系统应如何处理？
3. 分块时保留"制度名称"作为 metadata 有什么好处？
4. 如何评估这个 HR 问答系统的好坏？

### 自测题答案

1. **引用价值**：可追溯/审计、用户可验证、降低幻觉信任、便于更新知识库时定位来源。
2. **无关时处理**：设相似度阈值过滤；无命中返回"未找到"；或触发 fallback/人工；避免把无关上下文喂给 LLM。
3. **metadata 好处**：引用来源、按制度过滤检索、更新/删除某制度时精确定位、提升可追溯性。
4. **评估**：准确率/召回率、答案忠实度（faithfulness）、引用正确率、人工评测集、延迟/成本（L2-11）。

---

## 8. 延伸阅读

- LangChain RAG tutorial: https://python.langchain.com/docs/tutorials/rag/
- "Building RAG-based LLM Applications" (Anthropic)

---

## 9. 验收动作

- [ ] 跑通 `L2-05_hr_rag.py`，4 个问题均返回合理答案与引用
- [ ] 新增 3 条制度 + 2 个问题，验证泛化
- [ ] 故意问一个库中没有的问题，验证"不知道"降级

---

## 附录：FAQ

**Q：阈值设多高合适？**
A：从 cosine 0.5-0.7 起，用评测集调；太高漏答案，太低引入噪声。

**Q：引用要精确到句子吗？**
A：至少到 chunk/制度名；高合规场景（法律/医疗）需精确到段落。

**Q：制度更新了怎么办？**
A：只更新对应 chunk（增量），不必全库重建；保留版本号便于回溯。

**Q：为什么不用 LLM 直接读全文？**
A：上下文有限、成本高、易漏；检索先缩小范围再让 LLM 精读，质量与成本更优。
