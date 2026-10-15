"""
Task 5: 综合项目 — 智能客服 API 服务
目标：整合前 4 个任务，构建可运行的多轮对话 API
"""
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="智能客服 API", version="1.0.0")

# 初始化 LLM 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 对话存储（生产环境应使用 Redis/数据库）
conversations = {}

# 客服 System Prompt
CUSTOMER_SERVICE_PROMPT = """你是一个专业的电商客服助手，名叫"小智"。

职责：
1. 解答用户关于订单、物流、退换货、产品咨询等问题
2. 态度友好、耐心，使用礼貌用语
3. 对于无法解决的问题，引导用户联系人工客服
4. 每次回复控制在 100 字以内

特殊标记：
- 当问题超出能力范围时，在回复前加上 [需要: 人工]
- 当用户情绪激动时，先安抚情绪再解决问题

回复要求：
- 简洁明了，避免长篇大论
- 主动提供下一步操作建议
- 适当使用表情符号增加亲和力"""

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    need_human: bool = False
    token_usage: dict = {}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message
    
    # 获取或创建对话历史
    if session_id not in conversations:
        conversations[session_id] = []
    
    history = conversations[session_id]
    
    # 构建消息
    messages = [{"role": "system", "content": CUSTOMER_SERVICE_PROMPT}]
    messages.extend(history[-10:])  # 最近 10 条
    messages.append({"role": "user", "content": user_message})
    
    # 调用 LLM（带重试）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.5,
                max_tokens=300,
                timeout=30
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")
            time.sleep(2 ** attempt)
    
    reply = response.choices[0].message.content
    usage = response.usage
    
    # 更新历史
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    
    # 检测是否需要人工
    need_human = "[需要: 人工]" in reply
    reply = reply.replace("[需要: 人工]", "").strip()
    
    return ChatResponse(
        session_id=session_id,
        reply=reply,
        need_human=need_human,
        token_usage={
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(conversations),
        "total_messages": sum(len(h) for h in conversations.values())
    }

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in conversations:
        del conversations[session_id]
        return {"message": "会话已清除"}
    raise HTTPException(status_code=404, detail="会话不存在")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
