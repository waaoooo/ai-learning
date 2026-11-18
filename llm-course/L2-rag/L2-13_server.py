# -*- coding: utf-8 -*-
"""L2-13 LangChain + FastAPI 服务示例
运行：python L2-13_server.py  然后 curl -X POST http://127.0.0.1:8766/ask -d "年假怎么算"
"""
import os
from dotenv import load_dotenv
load_dotenv("../../week1/.env")

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=os.getenv("ZHIPU_API_KEY"),
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    temperature=0.3,
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是HR助手，回答简洁。"),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

app = FastAPI(title="L2-13 HR QA")

@app.post("/ask")
def ask(question: str):
    return {"answer": chain.invoke({"question": question})}

@app.get("/ask_stream")
def ask_stream(question: str):
    def gen():
        for chunk in chain.stream({"question": question}):
            yield chunk
    return StreamingResponse(gen(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8766)
