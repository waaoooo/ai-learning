# -*- coding: utf-8 -*-
"""L2-14 LangChain 链与 LCEL 综合实验"""
import os, re, urllib.request
from dotenv import load_dotenv

load_dotenv("../../week1/.env")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import (
    RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSequence,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=os.getenv("ZHIPU_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.3,
    max_tokens=512, timeout=30, max_retries=2,
)

# ---------- 1. 链的基本使用 ----------
print("=" * 70)
print("1. 链的基本使用：prompt | llm | parser")
prompt = ChatPromptTemplate.from_messages([("system", "你是HR助手，回答简洁。"), ("human", "{question}")])
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"question": "年假怎么算？"}))

# ---------- 2. RunnableLambda ----------
print("\n" + "=" * 70)
print("2. RunnableLambda：接入普通函数")
def word_count(text: str) -> int:
    return len(text)
wc = RunnableLambda(word_count)
print("字数:", wc.invoke("入职满1年享有5天年假"))
# 组合进链
chain2 = prompt | llm | StrOutputParser() | {"answer": RunnablePassthrough(), "len": wc}
print(chain2.invoke({"question": "年假怎么算？"}))

# ---------- 3. RunnableParallel + RunnablePassthrough ----------
print("\n" + "=" * 70)
print("3. RunnableParallel + RunnablePassthrough")
parallel = RunnableParallel(
    direct=RunnableLambda(lambda x: x["question"]) | llm,
    summary=prompt | llm | StrOutputParser(),
)
out = parallel.invoke({"question": "用一句话解释RAG"})
print("direct :", out["direct"].content[:60])
print("summary:", out["summary"][:60])

# ---------- 4. RunnableSequence 显式 ----------
print("\n" + "=" * 70)
print("4. RunnableSequence 显式组合")
seq = RunnableSequence(prompt, llm, StrOutputParser())
print(seq.invoke({"question": "报销流程是什么？"}))

# ---------- 5. 增强模板：FewShot + partial + Pydantic 解析 ----------
print("\n" + "=" * 70)
print("5. FewShotPromptTemplate + partial + PydanticOutputParser")
examples = [
    {"q": "年假怎么算？", "a": "满1年5天，满3年10天，满5年15天。"},
    {"q": "加班费？", "a": "工作日1.5倍，周末2倍，节假日3倍。"},
]
ex_prompt = PromptTemplate.from_template("问：{q}\n答：{a}")
few = FewShotPromptTemplate(
    examples=examples, example_prompt=ex_prompt,
    prefix="你是HR助手，参考示例：", suffix="问：{q}\n答：",
)
class QA(BaseModel):
    answer: str = Field(description="答案")
    source: str = Field(description="来源")
parser = PydanticOutputParser(pydantic_object=QA)
few_chain = few.partial(prefix="你是HR助手（partial 示例）") | llm | StrOutputParser()
try:
    print("FewShot:", few_chain.invoke({"q": "出差补贴？"})[:80])
except Exception as e:
    print("FewShot 失败:", e)

# 结构化输出：用 with_structured_output（function calling，最稳）
try:
    structured = llm.with_structured_output(QA)
    r = structured.invoke("资料：年假：满1年5天。问题：年假怎么算？来源：年假制度")
    print("Structured:", r.answer, "|", r.source)
except Exception as e:
    print("Structured 失败:", e)
# PydanticOutputParser 作为备选 + 健壮 JSON 提取
import json as _json

def extract_json(text: str) -> dict:
    """从模型输出中健壮提取第一个 JSON 对象（自定义解析器示例）"""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("未找到 JSON 对象")
    return _json.loads(m.group(0))

struct_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是HR助手。只输出JSON，格式：{{\"answer\":\"...\",\"source\":\"...\"}}。"),
    ("human", "资料：{ctx}\n问题：{q}"),
])
struct_chain = struct_prompt | llm | StrOutputParser()
try:
    raw = struct_chain.invoke({"ctx": "年假：满1年5天。", "q": "年假怎么算？"})
    print("原始输出:", raw[:120])
    obj = extract_json(raw)
    qa = QA(**obj)
    print("自定义解析器:", qa.answer, "|", qa.source)
except Exception as e:
    print("自定义解析失败:", str(e)[:120])

# ---------- 6. 多用户聊天机器人 ----------
print("\n" + "=" * 70)
print("6. RunnableWithMessageHistory：多用户聊天机器人")
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
u1a = multi.invoke({"input": "你好，我叫小明"}, config={"configurable": {"session_id": "u1"}})
u1b = multi.invoke({"input": "我叫什么？"}, config={"configurable": {"session_id": "u1"}})
u2a = multi.invoke({"input": "我叫什么？"}, config={"configurable": {"session_id": "u2"}})
print("u1 第一轮:", u1a[:50])
print("u1 第二轮:", u1b[:50])
print("u2（隔离）:", u2a[:50])

# ---------- 7. LCEL 读取在线网页 ----------
print("\n" + "=" * 70)
print("7. LCEL 读取在线网页")
def fetch_text(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", "ignore")
        text = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()[:1500]
    except Exception as e:
        return f"（网络不可用，使用本地兜底文本）RAG 是检索增强生成，用于让 LLM 基于外部知识回答问题。错误：{e}"

fetch = RunnableLambda(fetch_text)
web_chain = (
    {"content": fetch}
    | ChatPromptTemplate.from_messages([
        ("system", "用中文总结以下内容，不超过80字。"),
        ("human", "{content}"),
    ])
    | llm
    | StrOutputParser()
)
print("URL: https://example.com")
print("总结:", web_chain.invoke("https://example.com")[:120])

# ---------- 8. 链的元数据与输入输出 ----------
print("\n" + "=" * 70)
print("8. 链的元数据与输入输出")
print("类型:", type(chain).__name__)
print("steps:", [type(s).__name__ for s in chain.steps])
print("input_schema:", chain.input_schema.__name__)
print("output_schema:", chain.output_schema.__name__)
print("graph 节点:", [n.name for n in chain.get_graph().nodes.values()])
