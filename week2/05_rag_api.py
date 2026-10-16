"""
W2 综合：RAG 文档问答 API

接口：
  POST /ask    {"question": "..."}  →  检索 + 生成回答
  POST /eval                         →  跑评测集，返回准确率
  GET  /health                       →  健康检查

embedding 接口在 embed()，换神经模型只改这一个函数。
"""
import os
import re
import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

app = FastAPI(title="W2 RAG API", version="1.0.0")

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# ── 文档库 ──
DOCS = [
    "【住宿制度】一线城市住宿每晚不超过500元，其他城市不超过350元，超出部分自理。",
    "【交通制度】高铁限二等座，飞机限经济舱，市内打车需事前审批。",
    "【年假制度】正式员工年假15天，入职满一年后方可享受。",
    "【餐补制度】出差期间每人每天餐补100元，凭发票实报实销。",
    "【报销时限】费用发生后30天内提交报销，逾期不予报销。",
    "【加班制度】工作日加班支付1.5倍工资，周末2倍，法定节假日3倍。",
]

# ── 评测集 ──
EVAL_SET = [
    ("一线城市住宿标准是多少？", 0, "500"),
    ("出差坐高铁能坐什么座位？", 1, "二等座"),
    ("正式员工年假有几天？", 2, "15"),
    ("出差餐补每天多少钱？", 3, "100"),
    ("报销最晚什么时候提交？", 4, "30"),
    ("周末加班工资是几倍？", 5, "2"),
]


# ══════════════════════════════════════════════════════════
# ★ 唯一的切换点：换神经 embedding 只改这个函数 ★
# ══════════════════════════════════════════════════════════
def embed(texts: list[str]) -> np.ndarray:
    """文本 → 向量。当前：本地 n-gram TF-IDF。将来：换成神经模型。"""
    def tokenize(t):
        t = re.sub(r"\s+", "", t)
        return [t[i:i + n] for n in (1, 2) for i in range(len(t) - n + 1)]

    tokenized = [tokenize(t) for t in texts]
    vocab = {}
    for toks in tokenized:
        for t in set(toks):
            if t not in vocab:
                vocab[t] = len(vocab)
    V = len(vocab)
    df = np.zeros(V)
    for toks in tokenized:
        for t in set(toks):
            df[vocab[t]] += 1
    idf = np.log((1 + len(texts)) / (1 + df)) + 1
    M = np.zeros((len(texts), V))
    for i, toks in enumerate(tokenized):
        for t in toks:
            M[i, vocab[t]] += 1
        M[i] *= idf
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(norms == 0, 1, norms)


# ── 检索 ──
def retrieve(query: str, top_k: int = 1):
    vecs = embed(DOCS + [query])
    scores = [float(vecs[-1] @ dv) for dv in vecs[:-1]]
    rank = sorted(range(len(DOCS)), key=lambda i: scores[i], reverse=True)
    return rank[:top_k], scores


# ── 生成 ──
def answer(query: str, doc: str) -> str:
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model="glm-4.7-flash",
                messages=[
                    {"role": "system", "content": "你是制度问答助手，只能根据资料回答，不要编造。"},
                    {"role": "user", "content": f"资料：{doc}\n\n问题：{query}"},
                ],
                temperature=0.1,
                max_tokens=150,
                extra_body={"thinking": {"type": "disabled"}},
            ).choices[0].message.content
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


# ── 接口 ──
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved_doc: str
    score: float


SCORE_THRESHOLD = 0.15   # 低于此分视为"检索不可信"


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    top, scores = retrieve(req.question)
    doc = DOCS[top[0]]
    score = scores[top[0]]
    # 置信度门控：检索分太低就别硬答，避免胡编
    if score < SCORE_THRESHOLD:
        return AskResponse(
            question=req.question,
            answer="抱歉，资料库中没有找到相关信息，无法回答。",
            retrieved_doc=doc,
            score=round(score, 3),
        )
    ans = answer(req.question, doc)
    return AskResponse(
        question=req.question,
        answer=ans,
        retrieved_doc=doc,
        score=round(score, 3),
    )


@app.post("/eval")
async def eval():
    r_ok = a_ok = 0
    details = []
    for q, expect_idx, keyword in EVAL_SET:
        top, _ = retrieve(q)
        hit = DOCS[top[0]]
        ans = answer(q, hit)
        rok = top[0] == expect_idx
        aok = keyword in (ans or "")
        r_ok += rok
        a_ok += aok
        details.append({"q": q, "retrieval_ok": rok, "answer_ok": aok})
    n = len(EVAL_SET)
    return {
        "total": n,
        "retrieval_accuracy": round(r_ok / n, 3),
        "answer_accuracy": round(a_ok / n, 3),
        "details": details,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "docs": len(DOCS), "eval_questions": len(EVAL_SET)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
