# L2-14 · LangChain 的链和 LCEL

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：掌握 LangChain 链（Chain）与 LCEL 的核心组件（RunnableLambda / RunnableParallel / RunnablePassthrough / RunnableWithMessageHistory），能组装、组合并调试链，完成多用户聊天与网页读取两个实战。
> **先修**：L2-13（Model I/O）。
> **预计时长**：90-120 分钟（含实验）

---

## 1. 链（Chain）基本概念

### 1.1 什么是链
**链 = 把多个处理步骤串起来，前一步的输出是后一步的输入。** 在 LangChain 中，每个步骤都是一个 `Runnable`。

```
prompt | llm | parser
```

### 1.2 为什么用链
- 复用：把"提示→模型→解析"封装成一个对象
- 组合：像搭积木一样拼装复杂流程
- 标准化：统一 `invoke/stream/batch` 接口
- 可观测：链的每一步都可追踪

### 1.3 Runnable 协议
每个 Runnable 都实现：
| 方法 | 说明 |
|---|---|
| `invoke(input)` | 同步单次 |
| `stream(input)` | 流式输出 |
| `batch(inputs)` | 批量（并发） |
| `ainvoke/astream/abatch` | 异步版本 |

---

## 2. LCEL 深入

### 2.1 `|` 操作符与 RunnableSequence
`a | b` 等价于 `RunnableSequence(a, b)`：把两个 Runnable 顺序连接。

```python
chain = prompt | llm | StrOutputParser()
print(type(chain).__name__)   # RunnableSequence
print(chain.steps)            # [ChatPromptTemplate, ChatOpenAI, StrOutputParser]
```

### 2.2 组件详解

#### ① RunnableLambda：接入任意函数
```python
from langchain_core.runnables import RunnableLambda
wc = RunnableLambda(lambda t: len(t))
print(wc.invoke("你好"))   # 2
# 组合
chain = prompt | llm | StrOutputParser() | {"answer": RunnablePassthrough(), "len": wc}
```
- 注意：Lambda 的输入是**上一步的输出**；若要取字典字段，先做映射。

#### ② RunnableParallel：并行分支
```python
parallel = RunnableParallel(
    direct=RunnableLambda(lambda x: x["question"]) | llm,
    summary=prompt | llm | StrOutputParser(),
)
out = parallel.invoke({"question": "..."})
```
- 各分支**同时**执行，结果合并为 dict
- 注意：直接放 `llm` 会收到 dict 导致报错，需先用 Lambda 取出字段

#### ③ RunnablePassthrough：原样透传
```python
chain = prompt | llm | StrOutputParser() | {"answer": RunnablePassthrough(), "len": wc}
```
- 把上一步结果原封不动放进输出 dict，常用于"附加元信息"

#### ④ RunnableSequence：显式顺序
```python
seq = RunnableSequence(prompt, llm, StrOutputParser())
```
- 与 `|` 等价，但可读性更好，便于动态构造

#### ⑤ RunnableWithMessageHistory：多轮/多用户记忆
```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手，记住用户说过的话。"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
chat_chain = chat_prompt | llm | StrOutputParser()

store = {}
def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

multi = RunnableWithMessageHistory(
    chat_chain, get_history,
    input_messages_key="input", history_messages_key="history",
)
multi.invoke({"input": "我叫小明"}, config={"configurable": {"session_id": "u1"}})
multi.invoke({"input": "我叫什么？"}, config={"configurable": {"session_id": "u1"}})
```
- `session_id` 隔离不同用户/会话；`store` 是历史存储（生产换 Redis/DB）

### 2.3 组合模式
```
fan-out：一个输入 → 多个并行分支（RunnableParallel）
fan-in ：多个分支 → 合并（dict 合并 / RunnableLambda 归并）
嵌套  ：链里再放链
```

---

## 3. 增强提示词模板与输出解析器

### 3.1 FewShotPromptTemplate（少样本示例）
```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
ex = PromptTemplate.from_template("问：{q}\n答：{a}")
few = FewShotPromptTemplate(
    examples=[{"q": "年假怎么算？", "a": "满1年5天，满3年10天。"}],
    example_prompt=ex,
    prefix="你是HR助手，参考示例：", suffix="问：{q}\n答：",
)
```

### 3.2 模板部分格式化（partial）
```python
few.partial(prefix="你是HR助手（已固定前缀）")
```
- 预填部分变量，剩余变量运行时传入

### 3.3 输出解析器
- `StrOutputParser`：字符串（默认）
- `PydanticOutputParser`：按 schema 解析为对象
- 自定义解析器：从文本中健壮提取 JSON

```python
import json, re
def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("未找到 JSON")
    return json.loads(m.group(0))
```

### 3.4 结构化输出
```python
structured = llm.with_structured_output(QA)   # function calling，最稳
r = structured.invoke("...")
```
> ⚠️ 部分模型（如某些 GLM 版本）对 function calling 支持不稳定，此时用"prompt 约束 + 自定义 JSON 提取"兜底。

---

## 4. 实战一：多用户聊天机器人

**需求**：多个用户同时对话，各自记住自己的上下文，互不串扰。

**要点**：
- `RunnableWithMessageHistory` + `session_id`
- `MessagesPlaceholder("history")` 注入历史
- 历史存储用 dict（演示）→ 生产用 Redis/数据库

**验证**：用户 u1 说名字后能回忆；用户 u2 问同样问题应"不知道"（隔离）。

---

## 5. 实战二：LCEL 读取在线网页

**需求**：抓取网页 → 清洗 HTML → 总结。

```python
def fetch_text(url: str) -> str:
    html = urllib.request.urlopen(url, timeout=10).read().decode("utf-8", "ignore")
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:1500]

web_chain = (
    {"content": RunnableLambda(fetch_text)}
    | ChatPromptTemplate.from_messages([
        ("system", "用中文总结以下内容，不超过80字。"),
        ("human", "{content}"),
    ])
    | llm
    | StrOutputParser()
)
print(web_chain.invoke("https://example.com"))
```
- 网络不可用时给兜底文本，保证链可跑

---

## 6. 查看链的元数据与输入输出

```python
print(type(chain).__name__)                 # RunnableSequence
print([type(s).__name__ for s in chain.steps])
print(chain.input_schema)                   # 输入类型
print(chain.output_schema)                  # 输出类型
print([n.name for n in chain.get_graph().nodes.values()])  # 图节点
```
- 用于调试、文档生成、前端渲染

---

## 7. 调试与常见坑

| 坑 | 表现 | 修法 |
|---|---|---|
| 直接把 llm 放进 Parallel | 收到 dict 报错 | 先用 Lambda 取字段 |
| RunnableLambda 返回类型不匹配 | 下一步报错 | 明确输入/输出类型 |
| 历史不生效 | 每次都"不记得" | 检查 session_id 与 history_messages_key |
| JSON 解析失败 | 模型未按格式 | prompt 强约束 + 自定义提取 + 重试 |
| 链太长难调试 | 不知哪步出错 | `config={"verbose": True}` 或打印中间结果 |

---

## 8. 最小可运行示例

配套脚本：`L2-14_lcel.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-14_lcel.py
```

脚本演示：基础链、RunnableLambda、Parallel、Sequence、FewShot/partial、自定义 JSON 解析、多用户聊天、网页读取、链元数据。

---

## 9. 实验与练习

1. 用 RunnableParallel 同时跑"直答"和"RAG 答"，对比结果。
2. 给多用户聊天加"清空历史"功能（`store[sid].clear()`）。
3. 把网页读取链扩展为"抓取→分块→检索→总结"。
4. 打印一条复杂链的 `get_graph()`，画出节点关系。

---

## 10. 小结

- 链 = Runnable 的组合；`|` 生成 RunnableSequence。
- 核心组件：RunnableLambda（函数）、RunnableParallel（并行）、RunnablePassthrough（透传）、RunnableWithMessageHistory（记忆）。
- 模板与解析器是链的"输入/输出"端；自定义解析器提升健壮性。
- 元数据（steps/schema/graph）便于调试与文档化。

---

## 11. 自测题

1. `|` 操作符背后是什么？`a | b | c` 是什么类型？
2. RunnableParallel 中直接放 `llm` 为什么会报错？怎么修？
3. RunnablePassthrough 的典型用途？
4. RunnableWithMessageHistory 如何实现多用户隔离？需要哪些 key？
5. 为什么部分模型用 `with_structured_output` 会失败？如何兜底？

### 自测题答案

1. **背后**：`Runnable.__or__` 组合成 `RunnableSequence`；`a | b | c` 是 `RunnableSequence`，`steps` 为 [a, b, c]。
2. **报错原因**：Parallel 把输入 dict 原样传给 `llm`，而 chat 模型需要 str/消息列表；修法：先用 `RunnableLambda(lambda x: x["question"]) | llm`。
3. **Passthrough 用途**：把上一步结果原样放进输出结构（如附带长度、来源等元信息），实现 fan-in。
4. **多用户隔离**：通过 `config={"configurable": {"session_id": ...}}` 区分；需设置 `input_messages_key` 与 `history_messages_key`，并为每个 session 维护历史存储。
5. **失败与兜底**：部分模型 function calling 不稳定或返回空；兜底用 prompt 强约束输出 JSON + 正则提取 + Pydantic 校验 + 重试/换模型。

---

## 12. 延伸阅读

- LCEL 概念：https://python.langchain.com/docs/concepts/lcel/
- Runnable 协议：https://python.langchain.com/docs/concepts/runnables/
- LangChain "Chat memory" 文档

---

## 13. 验收动作

- [ ] 跑通 `L2-14_lcel.py`，观察 8 类链输出
- [ ] 实现多用户聊天并验证会话隔离
- [ ] 用 LCEL 完成网页抓取+总结
- [ ] 打印并解读一条链的 steps / schema / graph
