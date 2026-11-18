# -*- coding: utf-8 -*-
"""L2-13 LangChain Model I/O 综合实验"""
import os
from dotenv import load_dotenv

load_dotenv("../../week1/.env")

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import (
    PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, FewShotPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=os.getenv("ZHIPU_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.3,
    max_tokens=512,
    timeout=30,
    max_retries=2,
)

# ---------- 1. 消息类型 ----------
print("=" * 70)
print("1. 消息类型")
msgs = [
    SystemMessage(content="你是HR助手，只根据资料回答。"),
    HumanMessage(content="你好"),
]
resp = llm.invoke(msgs)
print("AIMessage:", resp.content[:80])
print("类型:", type(resp).__name__)

# ---------- 2. 提示模板 ----------
print("\n" + "=" * 70)
print("2. 提示模板")

# 2.1 PromptTemplate
tpl = PromptTemplate.from_template("用{language}写一个{description}的函数")
print("[PromptTemplate]", tpl.format(language="Python", description="快速排序"))

# 2.2 ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，回答要{constraint}。"),
    ("human", "{question}"),
])
print("[ChatPromptTemplate]", [m.content for m in chat_prompt.format_messages(
    role="HR助手", constraint="简洁", question="年假怎么算？")])

# 2.3 MessagesPlaceholder
mp_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])
print("[MessagesPlaceholder]", [m.content for m in mp_prompt.format_messages(
    history=[HumanMessage(content="你好")], question="继续")])

# 2.4 FewShotPromptTemplate
examples = [
    {"q": "年假怎么算？", "a": "按入职年限：1年5天，3年10天，5年15天。"},
    {"q": "加班费？", "a": "工作日1.5倍，周末2倍，节假日3倍。"},
]
example_prompt = PromptTemplate.from_template("问：{q}\n答：{a}")
few_shot = FewShotPromptTemplate(
    examples=examples, example_prompt=example_prompt,
    prefix="你是HR助手，参考示例回答：", suffix="问：{q}\n答：",
)
print("[FewShotPromptTemplate]\n" + few_shot.format(q="出差补贴？"))

# ---------- 3. 输出解析器 ----------
print("\n" + "=" * 70)
print("3. 输出解析器")

# 3.1 StrOutputParser + LCEL
chain = chat_prompt | llm | StrOutputParser()
print("[StrOutputParser]", chain.invoke({"role": "HR助手", "constraint": "一句话", "question": "年假怎么算？"}))

# 3.2 PydanticOutputParser
class HRAnswer(BaseModel):
    answer: str = Field(description="答案")
    source: str = Field(description="来源制度名")
    confidence: float = Field(description="置信度0-1")

parser = PydanticOutputParser(pydantic_object=HRAnswer)
struct_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是HR助手。仅根据资料回答，按指定JSON格式输出。\n{format_instructions}"),
    ("human", "资料：{context}\n问题：{question}"),
])
struct_chain = struct_prompt | llm | parser
try:
    r = struct_chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "context": "年假：入职满1年5天，满3年10天，满5年15天。",
        "question": "年假怎么算？",
    })
    print("[PydanticOutputParser]", r.answer, "| source:", r.source, "| conf:", r.confidence)
except Exception as e:
    print("[PydanticOutputParser] 解析失败:", e)

# ---------- 4. LCEL 组合 ----------
print("\n" + "=" * 70)
print("4. LCEL 组合")
parallel = RunnableParallel(
    direct=llm,
    rag=llm | StrOutputParser(),
)
try:
    out = parallel.invoke("用一句话解释RAG")
    print("[RunnableParallel] direct:", out["direct"].content[:60])
    print("[RunnableParallel] rag   :", out["rag"][:60])
except Exception as e:
    print("[RunnableParallel] 失败:", e)

# ---------- 5. 流式 ----------
print("\n" + "=" * 70)
print("5. 流式输出")
for chunk in llm.stream("用一句话介绍LangChain"):
    print(chunk.content, end="", flush=True)
print()
