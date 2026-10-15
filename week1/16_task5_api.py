"""
Task 5（综合）：智能客服 API
用法：
  uvicorn 16_task5_api:app --port 8000
然后另开终端：
  curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
       -d '{"session_id":"u1","message":"你好"}'
"""
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="W1 智能客服 API", version="1.0.0")

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

SYSTEM_PROMPT = """你是电商客服"小智"，职责：
1. 解答订单、物流、退换货、产品咨询
2. 态度友好，每次回复不超过 80 字
3. 无法解决时回复"这个需要人工客服帮您" """

# 每个会话的长期记忆（演示用，生产应存数据库）
SESSIONS: dict[str, dict] = {}


def get_memory(session_id: str) -> str:
    return SESSIONS.get(session_id, {}).get("memory", "")


def update_memory(session_id: str, user_msg: str, reply: str):
    """极简记忆：记住用户提到的关键词（生产环境用 LLM 抽取）"""
    SESSIONS.setdefault(session_id, {"history": [], "memory": ""})
    hist = SESSIONS[session_id]["history"]
    hist.append({"role": "user", "content": user_msg})
    hist.append({"role": "assistant", "content": reply})
    SESSIONS[session_id]["history"] = hist[-6:]  # 滑动窗口：最近 3 轮


def hard_check(reply: str) -> bool:
    """硬约束：不超过 80 字"""
    return len(reply) <= 80


def call_llm(session_id: str, user_msg: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT + f"\n【用户记忆】{get_memory(session_id)}"}]
    msgs += SESSIONS[session_id]["history"]
    msgs.append({"role": "user", "content": user_msg})

    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="glm-4.7-flash",
                messages=msgs,
                temperature=0.5,
                max_tokens=200,
                extra_body={"thinking": {"type": "disabled"}},
            )
            reply = r.choices[0].message.content
            # 代码兜底：超长了就截断
            return reply if hard_check(reply) else reply[:80]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    history_count: int


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    SESSIONS.setdefault(req.session_id, {"history": [], "memory": ""})
    reply = call_llm(req.session_id, req.message)
    update_memory(req.session_id, req.message, reply)
    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        history_count=len(SESSIONS[req.session_id]["history"]),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(SESSIONS)}


@app.post("/reset/{session_id}")
async def reset(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"message": f"会话 {session_id} 已清空"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
