# LLM 基础课 · 学习总纲（基于 AI_GUIDE_ALL.md）

> **目标**：系统掌握大语言模型（LLM）的核心原理与工程实践，从 L1 基础认知 → L2 RAG → L3 Agent → L4 微调/部署，最终能独立做 LLM 应用与垂直模型。
>
> **方式**：交互式、分阶段、可验证。每课一个可运行示例 + 一个验收动作。与现有 `Week1-12`（AI 应用开发轨道）并行，不互相覆盖。
>
> **受众**：已有 Python + LLM API 调用经验（完成 Week1-12），跳过语法，专注 AI 内容。

---

## 🗺️ 阶段与课时地图

### L1 · 基础篇（待做）
- L1-01 AI1.0→AI2.0 演进与大模型兴起
- L1-02 大模型与通用人工智能
- L1-03 主流大模型与应用产品
- L1-04 大模型赋能行业分析
- L1-05 发展趋势与挑战（Prompt / RAG / 微调）
- L1-06 LLM vs 传统 ML 差异
- L1-07 Transformer 架构
- L1-08 Token / 注意力 / 自注意力
- L1-09 预训练 / SFT / RLHF
- L1-10 Prompt 提示工程（TOT 24点实战）

### L2 · RAG 开发实战工坊（进行中）
- [x] L2-01 RAG 前言：LLM 缺陷 / RAG 定义 / 三大范式
- [x] L2-02 Naive RAG：文档加载与分块
- [x] L2-03 Embedding 向量化与相似度（cosine / L2）+ top-k
- [x] L2-04 向量数据库选型与实战（faiss）
- [x] L2-05 应用实战：HR 制度智能问答
- [x] L2-06 应用实战：混合检索医疗实体
- [x] L2-07 商业化 RAG 痛点分析
- [ ] L2-08 Advanced RAG：Pre-Retrieval（索引优化 / 查询优化）
- [ ] L2-09 Advanced RAG：混合检索 + Post-Retrieval（rerank / RAG-Fusion / 上下文压缩）
- [ ] L2-10 Advanced RAG 实战：金融助手 / 多模态图文检索
- [ ] L2-11 RAG 应用评估（Ragas / Trulens）
- [ ] L2-12 RAGFlow / FastGPT / Dify 剖析
- [ ] L2-13 LangChain 应用生命周期与 Model I/O
- [ ] L2-14 LangChain 链与 LCEL
- [ ] L2-15 LangChain 与 RAG
- [ ] L2-16 LangChain 工具调用
- [ ] L2-17 RAG 评估深入（指标 / 实战 / 2023 汽车赛）

### L3 · Agent 智能体架构设计（待做）
### L4 · 微调与私有化部署（待做）
### 专题集（LlamaIndex / MCP / GraphRAG / DeepSeek / A2A）（待做）

---

## 📊 当前进度

| 课时 | 主题 | 状态 |
|:---:|---|:---:|
| L2-01 | RAG 前言 | ✅ 已验收 |

---

## ⚙️ 环境

- Python 3.9+，依赖：`openai>=1.0`、`python-dotenv`、`numpy`、`tiktoken`、`faiss-cpu`、`pypdf`、`langchain>=0.3`（core）
- API：智谱 GLM（OpenAI 兼容），密钥放 `week1/.env`（`ZHIPU_API_KEY`），代码里 `load_dotenv` 从 `../week1/.env` 读取
- 运行：`cd llm-course/L2-rag && python L2-01_*.py`（如有配套脚本）

---

## 📁 目录

```
llm-course/
├── README.md
├── L1-foundations/   (待)
├── L2-rag/
│   ├── L2-01_rag-intro.md
│   └── ...
├── L3-agent/         (待)
├── L4-finetune-deploy/ (待)
└── topics/           (待)
```
