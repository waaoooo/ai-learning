# L2-02 · Naive RAG：文档加载与分块

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：掌握 RAG 知识库构建第一步——把原始文档（md/pdf/txt）加载并切成合适大小的 chunk，理解分块策略、参数权衡与结构化分块。
> **先修**：L2-01（RAG 全流程认知）。
> **预计时长**：90-120 分钟（含实验）

---

## 1. 为什么分块（Chunking）

LLM 上下文窗口有限，且检索粒度影响精度：

- **块太大**：检索噪声大、无关内容稀释答案、token 浪费
- **块太小**：上下文不完整、语义碎片化、丢失跨句信息
- **目标**：每个 chunk 是一个"**自包含的语义单元**"，大小适中（通常 200-800 tokens）

> 分块质量直接决定 RAG 上限——这是 L2-07 P2 的根源。

---

## 2. 文档加载（Document Loading）

| 来源 | 工具 | 注意点 |
|---|---|---|
| Markdown / TXT | 直接 read | 保留标题层级有助于后续分块 |
| PDF | `pypdf` / `PyMuPDF` | 扫描件需 OCR；表格/多栏布局易错乱 |
| 网页 | `BeautifulSoup` / `trafilatura` | 去导航/广告噪声 |
| Word/Excel | `python-docx` / `pandas` | 表格需特殊处理 |

**原则**：加载时尽量保留结构信息（标题、段落），比纯文本更有价值。

### 2.1 PDF 加载示例
```python
from pypdf import PdfReader
reader = PdfReader("report.pdf")
text = "\n".join(page.extract_text() or "" for page in reader.pages)
```
- 扫描件 `extract_text()` 为空 → 需 OCR（PyMuPDF + Tesseract / PaddleOCR）
- 表格建议单独抽取（`pdfplumber` / `camelot`）

---

## 3. 分块策略

| 策略 | 做法 | 适用 |
|---|---|---|
| **固定字符/Token** | 每 N 字切一刀 | 简单，但可能切断句子 |
| **递归字符分割**（推荐默认） | 优先按 `\n\n` → `\n` → `。` → 空格 → 字符递归切 | 通用，兼顾语义 |
| **按句子/段落** | 以句号/换行为界 | 短文本 |
| **语义分块** | embedding 相似度断点切分 | 高质量，成本高 |
| **结构化分块** | 按 Markdown 标题/HTML 标签切 | 文档有清晰结构时最佳 |

### 3.1 关键参数
- `chunk_size`：目标块大小（字符或 token），常用 **500-1000 字符**
- `chunk_overlap`：块间重叠（**10-20%**），防止信息在边界丢失
- `length_function`：计数函数（字符数 vs token 数）；token 更准确但需 tokenizer

### 3.2 权衡
- overlap 增大 → 召回更好但冗余/成本更高
- chunk_size 增大 → 上下文更完整但检索噪声更大

### 3.3 递归字符分块示例
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=80,
    separators=["\n## ", "\n### ", "\n\n", "\n", "。", " "],
    length_function=len,
)
chunks = splitter.split_text(text)
```

### 3.4 结构化分块（Markdown）
```python
from langchain_text_splitters import MarkdownHeaderTextSplitter
headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
docs = splitter.split_text(md_text)   # 每个 chunk 带 metadata
```

---

## 4. 分块效果评估

- **直观**：打印 chunk 数、平均长度、最长/最短
- **语义**：抽查 chunk 是否"自包含"（不看上下文能否理解）
- **检索**：对评测集看 Context Precision/Recall 随 chunk_size 变化（L2-11）
- **经验**：500-1000 字符 + 10-20% overlap 是通用起点

---

## 5. 常见坑

| 坑 | 表现 | 修法 |
|---|---|---|
| 表格被切碎 | 行列错位、数字丢失 | 表格单独抽取 |
| 标题与正文分离 | 检索到正文但不知属于哪节 | 结构化分块 + metadata |
| 扫描件无文本 | 检索为空 | OCR |
| overlap 过大 | 检索结果重复 | 控制在 10-20% |
| 按字符数而非 token | 中文/英文混排偏差 | 用 token 计数 |

---

## 6. 最小可运行示例

加载 `AI_GUIDE_ALL.md`，用 `RecursiveCharacterTextSplitter` 分块，打印统计与样例。

配套脚本：`L2-02_chunking.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-02_chunking.py
```

---

## 7. 实验与练习

1. 尝试 chunk_size=200/500/1000，对比 chunk 数与平均长度。
2. 对一份 PDF（可选）跑加载+分块，观察表格/多栏是否错乱。
3. 用 MarkdownHeaderTextSplitter 对 `AI_GUIDE_ALL.md` 分块，对比与递归分块的差异。
4. 抽查 10 个 chunk，人工判断是否"自包含"。

---

## 8. 小结

- 分块是 RAG 质量的第一道关；默认用递归字符分割 + 10-20% overlap。
- chunk_size/overlap 是超参，需按场景调优（L2-11 量化）。
- 加载时保留结构信息（标题）可显著提升后续检索。

---

## 9. 自测题

1. 为什么 chunk_overlap 能减少边界信息丢失？代价是什么？
2. 语义分块 vs 递归字符分块：各自优缺点？
3. 对一份 100 页的 PDF 手册，你会选多大的 chunk_size？为什么？
4. 如果文档是 Markdown 带标题，更好的分块方式是什么？

### 自测题答案

1. **overlap**：让相邻块共享一段文本，避免完整句子/概念被切断后只在块边界而检索时丢失；代价是冗余存储、token 成本增加、检索结果重复。
2. **语义 vs 递归**：语义分块按 embedding 相似度断点切，语义连贯、质量高，但计算成本高、慢；递归字符分块按分隔符递归切，快、通用、可控，但可能切在语义边界内。
3. **100 页手册**：通常 500-1000 字符（300-600 tokens），手册多为段落/步骤，中等块兼顾上下文完整与检索精度；配合 10-20% overlap。
4. **Markdown 带标题**：按标题结构化分块（`MarkdownHeaderTextSplitter` / 按 ##/### 切），保留标题作为 metadata，提升检索相关性与可追溯性。

---

## 10. 延伸阅读

- LangChain Text Splitters 文档：https://python.langchain.com/docs/concepts/text_splitters/
- "Chunking Strategies for LLM Applications" (Pinecone blog)
- LlamaIndex Node Parsers：https://docs.llamaindex.ai/

---

## 11. 验收动作

- [ ] 跑通 `L2-02_chunking.py`，输出 chunk 数与样例
- [ ] 尝试 chunk_size=200/500/1000，对比 chunk 数与平均长度
- [ ] 对一份 PDF（可选）跑加载+分块，观察表格/多栏是否错乱

---

## 附录：FAQ

**Q：chunk_size 用字符还是 token？**
A：token 更准（与模型上下文一致），但需 tokenizer；字符数简单快。中文 1 字≈1-2 token，可先按字符估算再换算。

**Q：overlap 设多大？**
A：10-20% 的 chunk_size。太小边界信息丢，太大冗余/成本升。

**Q：分块越细越好吗？**
A：不是。过细会切断语义、丢失上下文；需在 Precision 与 Recall 间权衡（L2-11）。

**Q：PDF 表格怎么处理最稳？**
A：`pdfplumber`/`camelot` 单独抽取表格，转 Markdown/结构化，再作为独立 chunk 并带表头 metadata。

**Q：为什么同一文档不同分块方式效果差很多？**
A：分块决定了检索粒度与语义完整性；建议用评测集量化（L2-11），不要凭感觉。
