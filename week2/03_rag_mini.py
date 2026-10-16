"""
W2 第 3+4 步：本地 embedding + 完整 RAG 链路（最小可运行）

说明：付费 embedding 不可用、HuggingFace 被墙，这里用「字符 n-gram TF-IDF」
模拟 embedding。真实项目换成 sentence-transformers / embedding API，
接口完全一致：文本 → 向量 → 余弦相似度 → 最近邻检索。

链路：文档库 → embedding → 检索 top1 → 拼进 prompt → LLM 生成
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

# ── 1. 文档库（模拟你公司的制度文档）──
DOCS = [
    "【住宿制度】一线城市住宿每晚不超过500元，其他城市不超过350元，超出部分自理。",
    "【交通制度】高铁限二等座，飞机限经济舱，市内打车需事前审批。",
    "【年假制度】正式员工年假15天，入职满一年后方可享受。",
]
QUERY = "一线城市住宿标准是多少？"


# ── 2. 本地 embedding：字符 n-gram + TF-IDF ──
def tokenize(text):
    text = re.sub(r"\s+", "", text)
    grams = []
    for n in (1, 2):                       # 1-gram + 2-gram
        for i in range(len(text) - n + 1):
            grams.append(text[i:i + n])
    return grams


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
    return M / np.where(norms == 0, 1, norms)   # L2 归一化


# ── 3. 检索：找最相似的 chunk ──
all_vecs = embed(DOCS + [QUERY])
doc_vecs, q_vec = all_vecs[:-1], all_vecs[-1]
scores = [float(q_vec @ dv) for dv in doc_vecs]      # 已归一化，点积=余弦
rank = sorted(range(len(DOCS)), key=lambda i: scores[i], reverse=True)

print("=" * 60)
print("文档相似度（余弦，越接近 1 越相关）")
print("=" * 60)
for i in rank:
    print(f"  {scores[i]:.3f}  {DOCS[i]}")

top = rank[0]
print(f"\n✅ 检索命中：{DOCS[top]}")


# ── 4. 生成：把检索到的内容拼进 prompt ──
def call_llm(messages):
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model="glm-4.7-flash",
                messages=messages,
                temperature=0.1,
                max_tokens=200,
                extra_body={"thinking": {"type": "disabled"}},
            ).choices[0].message.content
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


answer = call_llm([
    {"role": "system", "content": "你是制度问答助手，只能根据提供的资料回答，不要编造。"},
    {"role": "user", "content": f"资料：{DOCS[top]}\n\n问题：{QUERY}"},
])

print()
print("=" * 60)
print("RAG 生成的回答")
print("=" * 60)
print(answer)
print()
print("=" * 60)
print("对比：如果检索命中了【年假制度】，回答就会答非所问 ——")
print("所以「检索质量」直接决定「回答质量」，这是 RAG 的核心。")
print("=" * 60)
