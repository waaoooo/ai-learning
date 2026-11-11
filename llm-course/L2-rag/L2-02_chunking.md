# L2-02 · Naive RAG：文档加载与分块

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：掌握 RAG 知识库构建第一步——把原始文档（md/pdf/txt）加载并切成合适大小的 chunk，理解分块策略与参数权衡。
> **先修**：L2-01（RAG 全流程认知）。
> **预计时长**：45-60 分钟

---

## 1. 为什么分块（Chunking）

LLM 上下文窗口有限，且检索粒度影响精度：
- **块太大**：检索噪声大、无关内容稀释答案、token 浪费
- **块太小**：上下文不完整、语义碎片化、丢失跨句信息
- **目标**：每个 chunk 是一个"自包含的语义单元"，大小适中（通常 200-800 tokens）

---

## 2. 文档加载（Document Loading）

常见来源与加载器：

| 来源 | 工具 | 注意点 |
|---|---|---|
| Markdown / TXT | 直接 read | 保留标题层级有助于后续分块 |
| PDF | `pypdf` / `PyMuPDF` | 扫描件需 OCR；表格/多栏布局易错乱 |
| 网页 | `BeautifulSoup` / `trafilatura` | 去导航/广告噪声 |
| Word/Excel | `python-docx` / `pandas` | 表格需特殊处理 |

**原则**：加载时尽量保留结构信息（标题、段落），比纯文本更有价值。

---

## 3. 分块策略

| 策略 | 做法 | 适用 |
|---|---|---|
| **固定字符/Token** | 每 N 字切一刀 | 简单，但可能切断句子 |
| **递归字符分割**（推荐默认） | 优先按 `\n\n` → `\n` → `。` → 空格 → 字符 递归切 | 通用，兼顾语义 |
| **按句子/段落** | 以句号/换行为界 | 短文本 |
| **语义分块** | embedding 相似度断点切分 | 高质量，成本高 |
| **结构化分块** | 按 Markdown 标题/HTML 标签切 | 文档有清晰结构时最佳 |

**关键参数**：
- `chunk_size`：目标块大小（字符或 token），常用 500-1000 字符
- `chunk_overlap`：块间重叠（10-20%），防止信息在边界丢失
- `length_function`：计数函数（字符数 vs token 数）；token 更准确但需 tokenizer

**权衡**：overlap 增大 → 召回更好但冗余/成本更高；chunk_size 增大 → 上下文更完整但检索噪声更大。

---

## 4. 最小可运行示例

加载 `AI_GUIDE_ALL.md`，用 `RecursiveCharacterTextSplitter` 分块，打印统计与样例。

配套脚本：`L2-02_chunking.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-02_chunking.py
```

核心代码（节选）：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = open("../../AI_GUIDE_ALL.md", encoding="utf-8").read()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=80,
    separators=["\n## ", "\n### ", "\n\n", "\n", "。", " "],
    length_function=len,
)
chunks = splitter.split_text(text)
print("chunks:", len(chunks))
print(chunks[0][:300])
```

---

## 5. 小结

- 分块是 RAG 质量的第一道关；默认用递归字符分割 + 10-20% overlap。
- chunk_size/overlap 是超参，需按场景调优（L2-17 评估章节会讲如何量化）。
- 加载时保留结构信息（标题）可显著提升后续检索。

---

## 6. 自测题

1. 为什么 chunk_overlap 能减少边界信息丢失？代价是什么？
2. 语义分块 vs 递归字符分块：各自优缺点？
3. 对一份 100 页的 PDF 手册，你会选多大的 chunk_size？为什么？
4. 如果文档是 Markdown 带标题，更好的分块方式是什么？

## 自测题答案
- chunk_overlap 为什么减少边界信息丢失？代价？ 
overlap 让相邻块共享一段文本，避免完整句子/概念被切断后只出现在块边界而检索时丢失；代价是冗余存储、token 成本增加、检索结果重复。
- 语义分块 vs 递归字符分块 
语义分块按 embedding 相似度断点切，语义连贯、质量高，但计算成本高、慢；递归字符分块按分隔符递归切，快、通用、可控，但可能切在语义边界内。
- 100 页 PDF 手册 chunk_size 
通常 500-1000 字符（300-600 tokens），手册多为段落/步骤，中等块兼顾上下文完整与检索精度；过大会稀释、过小碎片化。配合 10-20% overlap。
- Markdown 带标题的更好分块 
按标题结构化分块（MarkdownHeaderTextSplitter / 按 ##/### 切），保留标题作为 metadata，提升检索相关性与可追溯性。

---

## 7. 延伸阅读

- LangChain Text Splitters 文档：https://python.langchain.com/docs/concepts/text_splitters/
- "Chunking Strategies for LLM Applications" (Pinecone blog)
- LlamaIndex Node Parsers：https://docs.llamaindex.ai/

---

## 8. 验收动作

- [ ] 跑通 `L2-02_chunking.py`，输出 chunk 数与样例
- [ ] 尝试 chunk_size=200/500/1000，对比 chunk 数与平均长度
- [ ] 对一份 PDF（可选）跑加载+分块，观察表格/多栏是否错乱
