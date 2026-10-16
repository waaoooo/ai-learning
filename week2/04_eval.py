"""
W2 第 5 步：RAG 评测 —— 用数字量化检索和回答质量

评测集 = 一批「已知正确答案」的问题，让 RAG 跑一遍，看它对不对。
两个指标：
  1. 检索准确率：有没有命中正确的文档
  2. 回答准确率：答案里有没有包含正确信息
"""
import os
import re
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

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

# ── 评测集：(问题, 应命中的文档索引, 答案应包含的关键词) ──
EVAL_SET = [
    ("一线城市住宿标准是多少？",        0, "500"),
    ("出差坐高铁能坐什么座位？",        1, "二等座"),
    ("正式员工年假有几天？",            2, "15"),
    ("出差餐补每天多少钱？",            3, "100"),
    ("报销最晚什么时候提交？",          4, "30"),
    ("周末加班工资是几倍？",            5, "2"),
]


# ── embedding（同第 3 步）──
def tokenize(text):
    text = re.sub(r"\s+", "", text)
    return [text[i:i + n] for n in (1, 2) for i in range(len(text) - n + 1)]


def embed(texts):
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


def retrieve(query, top_k=1):
    vecs = embed(DOCS + [query])
    scores = [float(vecs[-1] @ dv) for dv in vecs[:-1]]
    rank = sorted(range(len(DOCS)), key=lambda i: scores[i], reverse=True)
    return rank[top_k - 1], scores


def answer(query, doc):
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


def run_eval(eval_set, label):
    print("=" * 70)
    print(f"【{label}】{len(eval_set)} 题")
    print("=" * 70)
    r_ok_n = a_ok_n = 0
    for q, expect_idx, keyword in eval_set:
        hit_idx, _ = retrieve(q)
        hit_doc = DOCS[hit_idx]
        ans = answer(q, hit_doc)
        r_ok = (hit_idx == expect_idx)
        a_ok = (keyword in (ans or ""))
        r_ok_n += r_ok
        a_ok_n += a_ok
        print(f"\n问题: {q}")
        print(f"  应命中: {DOCS[expect_idx][:18]}...  |  实际命中: {hit_doc[:18]}...")
        print(f"  检索: {'✅' if r_ok else '❌'}   回答: {'✅' if a_ok else '❌'}  {ans}")
    print(f"\n  {label} 检索准确率: {r_ok_n}/{len(eval_set)} = {r_ok_n/len(eval_set)*100:.0f}%")
    print(f"  {label} 回答准确率: {a_ok_n}/{len(eval_set)} = {a_ok_n/len(eval_set)*100:.0f}%\n")


# 简单集：问题和文档用词重合（lexical 能答对）
run_eval(EVAL_SET, "简单集（词面重合）")

# 困难集：同义改写（lexical 会失败 —— 这就是评测的价值）
HARD_EVAL_SET = [
    ("住酒店一晚最多能报多少？", 0, "500"),
    ("出差坐火车能坐什么位置？", 1, "二等座"),
    ("一年有多少天不用上班？", 2, "15"),
    ("出差吃饭能补多少钱？", 3, "100"),
]
run_eval(HARD_EVAL_SET, "困难集（同义改写）")

print("=" * 70)
print("结论：简单集 100% 给你虚假的自信；困难集暴露真实短板。")
print("评测的意义 = 在你上线前，用数字告诉你哪里不行。")
print("=" * 70)
