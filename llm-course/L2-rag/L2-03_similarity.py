# -*- coding: utf-8 -*-
"""L2-03 Embedding 相似度：cosine / L2 / top-k 演示"""
import os, hashlib, numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../../week1/.env")
client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4")

docs = [
    "公司年假：入职满1年享有5天年假，满3年10天，满5年15天。",
    "报销流程：员工提交发票→直属经理审批→财务复核→3个工作日到账。",
    "加班补偿：工作日加班按1.5倍，周末2倍，法定节假日3倍。",
    "体检福利：每年一次免费体检，入职满1年可带家属。",
]

def _local_embed(texts, dim=256):
    out = []
    for t in texts:
        v = np.zeros(dim, dtype="float32")
        for i in range(len(t)):
            for n in (1, 2, 3):
                if i + n <= len(t):
                    gram = t[i:i + n]
                    h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                    v[h % dim] += 1.0
        nrm = np.linalg.norm(v)
        if nrm > 0: v /= nrm
        out.append(v)
    return np.array(out, dtype="float32")

def embed(texts):
    try:
        r = client.embeddings.create(model="embedding-3", input=texts)
        V = np.array([d.embedding for d in r.data], dtype="float32")
        # 归一化，使内积 == cosine
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        return V / np.clip(norms, 1e-12, None)
    except Exception as e:
        print(f"[warn] 真实 embedding 不可用（{e}），使用本地 fallback。")
        return _local_embed(texts)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def l2(a, b):
    return float(np.linalg.norm(a - b))

def main():
    V = embed(docs)
    q = "年假怎么算？"
    qv = embed([q])[0]

    print("查询:", q)
    print("\n--- 各文档相似度 ---")
    rows = []
    for i, d in enumerate(docs):
        rows.append((i, cosine(qv, V[i]), l2(qv, V[i]), d))
    for i, c, l, d in sorted(rows, key=lambda x: -x[1]):
        print(f"[{i}] cosine={c:.4f}  L2={l:.4f}  {d[:40]}")

    # top-k
    k = 2
    scores = V @ qv  # 归一化后内积 == cosine
    top = np.argsort(scores)[::-1][:k]
    print(f"\n--- top-{k} ---")
    for idx in top:
        print(f"[{idx}] score={scores[idx]:.4f}  {docs[idx]}")

if __name__ == "__main__":
    main()
