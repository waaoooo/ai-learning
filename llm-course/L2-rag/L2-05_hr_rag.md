# L2-05 · 应用实战：HR 制度智能问答（向量检索）

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：把 L2-01~L2-04 串起来，实现一个端到端的 HR 制度问答系统：文档 → 分块 → embedding → 索引 → 检索 → 生成（带引用）。
> **先修**：L2-01 ~ L2-04。
> **预计时长**：60-90 分钟

---

## 1. 系统架构

```
HR 制度文档（md/txt）
   ↓  加载 + 分块 (L2-02)
chunks[]
   ↓  embedding (L2-03)
vectors
   ↓  faiss IndexFlatIP (L2-04)
index
   ↓  top-k 检索
相关 chunks + 来源
   ↓  prompt 拼接
LLM 生成答案（带引用）
```

---

## 2. 关键工程点

- **分块**：按制度条目切分，保留"制度名称"作为 metadata，便于引用
- **检索**：top-k=3~5；可加相似度阈值过滤低相关结果
- **Prompt**：明确"仅基于资料回答，不足则说不知道，附引用"
- **引用**：答案末尾列出来源 chunk id / 制度名
- **降级**：无相关结果时返回"未找到"，避免幻觉

---

## 3. 最小可运行示例

配套脚本：`L2-05_hr_rag.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-05_hr_rag.py
```

脚本内置 8 条 HR 制度，对 4 个测试问题跑问答，打印检索片段与答案。

---

## 4. 小结

- 端到端 RAG = 加载 → 分块 → embedding → 索引 → 检索 → 生成；每步都是 L2-01~04 的组合。
- 引用与"不知道"降级是生产 RAG 的基本素养。
- 本例用内存语料 + faiss，生产需持久化索引与文档库。

---

## 5. 自测题

1. 为什么 HR 问答要"附引用"？对用户和审计的价值？
2. 如果检索到的 top-3 都与问题无关，系统应如何处理？
3. 分块时保留"制度名称"作为 metadata 有什么好处？
4. 如何评估这个 HR 问答系统的好坏？（引出 L2-11 评估）

---

## 6. 延伸阅读

- LangChain RAG tutorial: https://python.langchain.com/docs/tutorials/rag/
- "Building RAG-based LLM Applications" (Anthropic)

---

## 7. 验收动作

- [ ] 跑通 `L2-05_hr_rag.py`，4 个问题均返回合理答案与引用
- [ ] 新增 3 条制度 + 2 个问题，验证泛化
- [ ] 故意问一个库中没有的问题，验证"不知道"降级
