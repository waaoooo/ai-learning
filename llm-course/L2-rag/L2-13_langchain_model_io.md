# L2-13 · LangChain 应用生命周期与 Model I/O

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：系统掌握 LangChain 的体系结构与开发生命周期，深入理解 Model I/O（消息、聊天模型、提示模板、输出解析器），并能用 LCEL 组装一条可运行的链。
> **先修**：L2-01 ~ L2-12；Python + OpenAI 兼容 API 调用经验。
> **预计时长**：90-120 分钟（含实验）
> **学习路径**：概念（1-3）→ Model I/O 四件套（4）→ LCEL 链（5）→ 调试与发布（6-7）→ 综合实验（8）→ 自测（11）

---

## 1. LangChain 是什么

### 1.1 定义
LangChain 是一个用于构建 **LLM 应用** 的开源框架，把"模型调用、提示管理、数据检索、工具编排、记忆、评估、部署"抽象成可组合的组件，让你用统一接口拼出复杂应用。

### 1.2 为什么需要它（LLM 应用的 7 个痛点）

| 痛点 | 没有框架时 | LangChain 提供 |
|---|---|---|
| 模型接口不统一 | 每家 SDK 不同 | `BaseChatModel` 统一接口 |
| 提示词难维护 | 字符串拼接散落各处 | PromptTemplate 模板化 |
| 输出格式不稳定 | 正则/try-except 解析 | OutputParser 结构化 |
| 多步编排混乱 | 手写嵌套函数 | Chain / LCEL 声明式编排 |
| 上下文/记忆缺失 | 自己存 history | Memory 组件 |
| 工具调用耦合 | 手写 function calling 循环 | Agent / Tool 抽象 |
| 调试困难 | 无追踪 | Callbacks / LangSmith |

### 1.3 版本演进（重要）

| 版本 | 变化 | 影响 |
|---|---|---|
| v0.1 | 引入 LCEL、Runnable | 声明式链成为核心 |
| v0.2 | 包结构拆分（langchain-core / -community / -openai） | 依赖更清晰 |
| v0.3 | 全面拥抱 LCEL，弃用旧 `LLMChain` 等 | 新代码一律用 LCEL |

> 本课基于 **langchain-core 0.3.x + langchain-openai 0.2.x**（课程环境）。

### 1.4 生态分层

```
┌─────────────────────────────────────────────┐
│ LangSmith   可观测性 / 评测 / Prompt 管理     │
├─────────────────────────────────────────────┤
│ LangGraph   有状态多步 Agent / 工作流（L3）   │
├─────────────────────────────────────────────┤
│ LangChain   组件 + 链（Model I/O/Retrieval…） │
├─────────────────────────────────────────────┤
│ 模型 Provider（OpenAI / GLM / Anthropic…）   │
└─────────────────────────────────────────────┘
```

---

## 2. LangChain 体系（核心组件）

LangChain 由 6 大模块组成，本课重点在 **Model I/O**：

| 模块 | 职责 | 关键类 |
|---|---|---|
| **Model I/O** | 与模型交互：消息、提示、解析 | `ChatOpenAI`, `ChatPromptTemplate`, `PydanticOutputParser` |
| **Retrieval** | 文档加载、切分、向量检索 | `DocumentLoader`, `TextSplitter`, `VectorStore` |
| **Chains** | 多步编排 | `RunnableSequence`, `RunnableParallel` |
| **Memory** | 跨轮/跨会话状态 | `ConversationBufferMemory` 等 |
| **Agents** | 自主决策与工具调用 | `AgentExecutor`（L3 详讲） |
| **Callbacks** | 日志/追踪/监控 | `ConsoleCallbackHandler`, LangSmith |

**数据流**：
```
输入 → Prompt → ChatModel → OutputParser → 输出
```

---

## 3. LangChain 应用生命周期（7 阶段）

一个生产级 LangChain 应用的开发闭环：

1. **需求定义**：输入/输出/边界/评估指标（对应 L2-07 痛点诊断）
2. **数据工程**：加载、清洗、切分、索引（L2-02）
3. **模型选型**：模型/参数/成本（本课 4.2）
4. **提示工程**：模板设计、few-shot（本课 4.3）
5. **检索接入**：RAG 管线（L2-03~11）
6. **链编排**：LCEL 组装（本课 5）
7. **评估与发布**：离线评测 + 线上监控（本课 6-7）

> 记住：**80% 的 RAG 质量取决于数据工程（第 2 步）和提示（第 4 步），而非框架本身。**

---

## 4. Model I/O 深入

Model I/O 是 LangChain 的"输入-输出"层，包含四个核心概念。

### 4.1 消息模型（Message Types）

LLM API 的输入不是纯字符串，而是**消息列表**。LangChain 用类型化消息表示：

| 类型 | 角色 | 用途 |
|---|---|---|
| `SystemMessage` | system | 设定身份/规则 |
| `HumanMessage` | user | 用户输入 |
| `AIMessage` | assistant | 模型回复 |
| `ToolMessage` | tool | 工具执行结果 |
| `FunctionMessage` | function | 旧版函数调用结果 |

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

msgs = [
    SystemMessage(content="你是HR助手，只根据资料回答。"),
    HumanMessage(content="年假怎么算？"),
]
```

**要点**：
- 多轮对话 = 在列表里追加 `HumanMessage` / `AIMessage`
- `ToolMessage` 用于把工具结果回传（L2-16）
- 消息可序列化，便于存入 Memory

### 4.2 聊天模型（ChatOpenAI）

`ChatOpenAI` 是 LangChain 对 OpenAI 兼容接口的封装，可接 GLM 等国产模型。

```python
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv("../../week1/.env")

llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=os.getenv("ZHIPU_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.3,
    max_tokens=512,
    timeout=30,
    max_retries=2,
)
```

**关键参数**：

| 参数 | 作用 | 建议 |
|---|---|---|
| `model` | 模型名 | 按任务选（flash 快、plus 强） |
| `temperature` | 随机性 0-2 | 事实问答 0-0.3；创意 0.7-1.0 |
| `top_p` | 核采样 | 一般不调，与 temperature 二选一 |
| `max_tokens` | 最大输出 | 按任务设，控成本 |
| `seed` | 随机种子 | 可复现（部分模型支持） |
| `stop` | 停止序列 | 控制输出边界 |
| `timeout` / `max_retries` | 稳定性 | 生产必设 |
| `model_kwargs` | 透传 provider 参数 | 如 `response_format` |
| `streaming` | 流式 | 提升体验（见 4.2.2） |

**调用方式**：

```python
# 1) invoke：单轮
resp = llm.invoke([HumanMessage(content="你好")])
print(resp.content)

# 2) batch：批量（并发）
llm.batch(["问题1", "问题2"])

# 3) stream：流式输出
for chunk in llm.stream("讲个笑话"):
    print(chunk.content, end="", flush=True)
```

**结构化输出**（用 Pydantic 定义 schema，见 4.4.3）：
```python
from pydantic import BaseModel
class Answer(BaseModel):
    answer: str
    confidence: float
structured = llm.with_structured_output(Answer)
```

### 4.3 提示模板（Prompt Templates）

模板把"可变部分"与"固定指令"分离，是提示工程的基础设施。

#### 4.3.1 PromptTemplate（单字符串）
```python
from langchain_core.prompts import PromptTemplate
tpl = PromptTemplate.from_template("用{language}写一个{description}的函数")
print(tpl.format(language="Python", description="快速排序"))
```

#### 4.3.2 ChatPromptTemplate（消息序列，推荐）
```python
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，回答要{constraint}。"),
    ("human", "{question}"),
])
messages = prompt.format_messages(role="HR助手", constraint="简洁", question="年假怎么算？")
```

#### 4.3.3 MessagesPlaceholder（注入历史消息）
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手。"),
    MessagesPlaceholder(variable_name="history"),  # 运行时注入
    ("human", "{question}"),
])
prompt.format_messages(history=[HumanMessage(content="你好")], question="继续")
```

#### 4.3.4 FewShotPromptTemplate（少样本）
```python
from langchain_core.prompts import FewShotPromptTemplate
examples = [
    {"q": "年假怎么算？", "a": "按入职年限：1年5天，3年10天，5年15天。"},
    {"q": "加班费？", "a": "工作日1.5倍，周末2倍，节假日3倍。"},
]
example_prompt = PromptTemplate.from_template("问：{q}\n答：{a}")
few_shot = FewShotPromptTemplate(
    examples=examples, example_prompt=example_prompt,
    prefix="你是HR助手，参考示例回答：", suffix="问：{q}\n答：",
)
print(few_shot.format(q="出差补贴？"))
```

**模板组合**：`prompt.partial(role="HR助手")` 可预填部分变量；`prompt | llm` 直接接模型（LCEL）。

### 4.4 输出解析器（Output Parsers）

把模型的自然语言输出转成程序可用的结构。

| 解析器 | 输出 | 适用 |
|---|---|---|
| `StrOutputParser` | 字符串 | 默认 |
| `JsonOutputParser` | dict/list | JSON 文本 |
| `PydanticOutputParser` | Pydantic 对象 | 强类型结构化 |
| 自定义 | 任意 | 特殊格式 |

#### 4.4.1 StrOutputParser
```python
from langchain_core.output_parsers import StrOutputParser
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"question": "你好"}))
```

#### 4.4.2 JsonOutputParser
```python
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser()
chain = prompt | llm | parser
result = chain.invoke({"question": "列出3种请假类型，JSON格式"})
```

#### 4.4.3 PydanticOutputParser（结构化，最实用）
```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class HRAnswer(BaseModel):
    answer: str = Field(description="答案")
    source: str = Field(description="来源制度名")
    confidence: float = Field(description="置信度0-1")

parser = PydanticOutputParser(pydantic_object=HRAnswer)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是HR助手。仅根据资料回答，按指定JSON格式输出。\n{format_instructions}"),
    ("human", "资料：{context}\n问题：{question}"),
])
chain = prompt | llm | parser
result = chain.invoke({
    "format_instructions": parser.get_format_instructions(),
    "context": "年假：1年5天，3年10天。",
    "question": "年假怎么算？",
})
print(result.answer, result.source, result.confidence)
```

> `get_format_instructions()` 会把 Pydantic schema 变成模型能理解的格式说明，是结构化输出的关键。

---

## 5. 链（Chain）与 LCEL 基础

### 5.1 LCEL 是什么
LCEL（LangChain Expression Language）用 `|` 管道把组件串成链，每个组件都是 `Runnable`（可 invoke/stream/batch）。

```
prompt | llm | parser
```

等价于：`parser.invoke(llm.invoke(prompt.invoke(input)))`。

### 5.2 常用 Runnable 组合

| 组合 | 写法 | 用途 |
|---|---|---|
| 顺序 | `a | b | c` | 管道 |
| 并行 | `RunnableParallel(x=a, y=b)` | 同时算多个 |
| 透传 | `RunnablePassthrough()` | 原样传递 |
| 自定义 | `RunnableLambda(fn)` | 接普通函数 |

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | llm
    | StrOutputParser()
)
print(chain.invoke({"question": "用一句话解释RAG"}))
```

### 5.3 链的调用方式
```python
chain.invoke(input)          # 同步
chain.batch([i1, i2])        # 批量
chain.stream(input)          # 流式
await chain.ainvoke(input)   # 异步
```

---

## 6. 跟踪与调试

### 6.1 verbose 模式
```python
chain.invoke(input, config={"verbose": True})  # 打印每步输入输出
```

### 6.2 Callbacks
```python
from langchain_core.callbacks import ConsoleCallbackHandler
chain.invoke(input, config={"callbacks": [ConsoleCallbackHandler()]})
```

### 6.3 LangSmith（可选）
设置环境变量后自动追踪每次调用：
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=...
export LANGCHAIN_PROJECT=llm-course
```
可在 LangSmith 后台看链路耗时、token、错误。

### 6.4 常见问题排查

| 现象 | 可能原因 | 排查 |
|---|---|---|
| 输出格式错 | prompt 未给格式说明 | 用 PydanticOutputParser + format_instructions |
| 答案幻觉 | 上下文不足 | 检查检索 chunks 与分数 |
| 链不生效 | 组件类型不匹配 | 打印中间结果 |
| 超时 | 模型/网络 | 设 timeout/max_retries |

---

## 7. 发布应用程序

### 7.1 用 FastAPI 封装链
```python
from fastapi import FastAPI
from langchain_core.runnables import RunnableLambda

app = FastAPI()
chain = prompt | llm | StrOutputParser()

@app.post("/ask")
def ask(question: str):
    return {"answer": chain.invoke({"question": question})}
```

### 7.2 流式输出
```python
from fastapi.responses import StreamingResponse

@app.get("/ask_stream")
def ask_stream(question: str):
    def gen():
        for chunk in chain.stream({"question": question}):
            yield chunk
    return StreamingResponse(gen(), media_type="text/plain")
```

### 7.3 服务化要点
- **并发**：异步链（`ainvoke`）避免阻塞
- **超时与重试**：模型调用设 timeout/max_retries
- **限流与成本**：记录 token，按用户限流
- **可观测**：接入 LangSmith / 日志
- **安全**：输入校验、prompt 注入防护（L2 安全相关）

---

## 8. 最小可运行示例（综合实验）

配套脚本：`L2-13_model_io.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-13_model_io.py
```

脚本依次演示：
1. 消息类型与 ChatOpenAI 调用
2. ChatPromptTemplate / MessagesPlaceholder / FewShotPromptTemplate
3. StrOutputParser / PydanticOutputParser 结构化输出
4. LCEL 链 `prompt | llm | parser`（含 stream）
5. FastAPI 服务代码（`L2-13_server.py`，可选启动）

> ⚠️ 若 embedding/模型账户无余额，脚本使用本地 fallback；Chat 使用 GLM。

---

## 9. 实验与练习

1. **参数实验**：同一问题分别用 temperature=0 / 0.7 / 1.5，观察输出差异，记录哪个适合事实问答。
2. **模板实验**：把 HR 问答 prompt 改成 ChatPromptTemplate + MessagesPlaceholder，注入 3 轮历史，观察多轮效果。
3. **结构化输出实验**：定义一个 `PolicyAnswer`（answer/source/confidence），用 PydanticOutputParser 跑 5 个问题，统计解析成功率。
4. **链实验**：用 RunnableParallel 同时跑"直答"和"RAG 答"，对比结果。
5. **调试实验**：打开 verbose，观察链每一步的输入输出。

---

## 10. 小结

- LangChain 用统一抽象解决 LLM 应用的 7 大痛点；核心是 Model I/O + Retrieval + Chains。
- 应用生命周期 7 阶段：需求→数据→模型→提示→检索→链→评估发布。
- Model I/O 四件套：消息、聊天模型、提示模板、输出解析器。
- LCEL 用 `|` 组装链，Runnable 支持 invoke/stream/batch/异步。
- 生产要关注调试（callbacks/LangSmith）、发布（FastAPI + 流式 + 限流）。

---

## 11. 自测题

1. LangChain 的 6 大模块是什么？本课重点哪个？
2. ChatOpenAI 相比直接 openai SDK 多给了什么能力？
3. PromptTemplate 与 ChatPromptTemplate 的区别？
4. PydanticOutputParser 为什么需要 `get_format_instructions()`？
5. LCEL 的 `|` 操作符背后是什么机制？
6. 生产环境发布 LangChain 应用要考虑哪些非功能需求？

### 自测题答案

1. **6 大模块**：Model I/O、Retrieval、Chains、Memory、Agents、Callbacks；本课重点 **Model I/O**。
2. **ChatOpenAI 多给的能力**：统一模型接口、消息抽象、与 LCEL 组件组合、结构化输出（with_structured_output）、回调/追踪集成；可直接接任意 OpenAI 兼容端点。
3. **PromptTemplate vs ChatPromptTemplate**：前者生成单个字符串；后者生成**消息列表**（可含 system/human/placeholder），更贴合 chat 模型。
4. **为什么需要 format_instructions**：把 Pydantic schema 转成模型可读的格式说明注入 prompt，引导模型输出符合 schema 的 JSON，否则解析易失败。
5. **`|` 背后机制**：`Runnable.__or__` 把两个 Runnable 组合成 `RunnableSequence`，输入依次流经各组件；每个组件都实现 `invoke/stream/batch`。
6. **非功能需求**：并发（异步）、超时重试、限流与成本控制、可观测（日志/LangSmith）、安全（输入校验/注入防护）、流式体验。

---

## 12. 延伸阅读

- LangChain 官方文档：https://python.langchain.com/docs/
- LCEL 概念：https://python.langchain.com/docs/concepts/lcel/
- LangSmith：https://smith.langchain.com

---

## 13. 验收动作

- [ ] 跑通 `L2-13_model_io.py`，观察 4 类 Model I/O 输出
- [ ] 用 PydanticOutputParser 完成 5 次结构化问答，记录成功率
- [ ] 用 FastAPI 启动 `L2-13_server.py` 并 curl 一次（可选）
- [ ] 完成第 9 节实验 1-3，记录实验结论
