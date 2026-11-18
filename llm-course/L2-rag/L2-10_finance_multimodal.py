# -*- coding: utf-8 -*-
"""L2-10 金融助手 RAG + 多模态 RAG（方式①/③）"""
import os, hashlib, math, numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../../week1/.env")
client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4")

# ---------------- 金融语料 ----------------
finance_docs = [
    ("2024年报", "公司2024年营收120亿元，同比增长15%；净利润18亿元，净利率15%。"),
    ("风险提示", "主要风险：行业竞争加剧、原材料价格波动、汇率波动。请投资者注意风险。"),
    ("研报摘要", "研报：给予'买入'评级，目标价25元，对应2025年PE 18倍。"),
    ("合规法规", "信息披露规则：重大合同需在2个交易日内公告，否则视为违规。"),
    ("分红公告", "2024年度分红：每10股派发现金红利3元，股权登记日2025-06-10。"),
    ("业务构成", "业务构成：To B业务占70%，To C业务占30%；海外收入占比20%。"),
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
        V /= np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
        return V
    except Exception:
        return _local_embed(texts)

def bm25_scores(corpus, q):
    qset = set(q)
    scores = []
    for d in corpus:
        dset = set(d)
        inter = len(qset & dset)
        scores.append(inter / math.sqrt(len(qset) * len(dset) + 1e-9))
    return np.array(scores)

def finance_rag(q, k=3):
    texts = [d[1] for d in finance_docs]
    V = embed(texts)
    qv = embed([q])[0]
    vec = V @ qv
    kw = bm25_scores(texts, q)
    # 加权融合（归一化）
    vec_n = (vec - vec.min()) / (vec.max() - vec.min() + 1e-9)
    kw_n = (kw - kw.min()) / (kw.max() - kw.min() + 1e-9)
    fused = 0.6 * vec_n + 0.4 * kw_n
    top = list(np.argsort(fused)[::-1][:k])
    return [(fused[i], finance_docs[i][0], finance_docs[i][1]) for i in top]

def main():
    print("=" * 70)
    print("【金融助手 RAG】")
    q = "2024年营收和净利润是多少？"
    print("Q:", q)
    hits = finance_rag(q)
    print("检索(混合+融合):")
    for s, name, text in hits:
        print(f"  score={s:.3f} [{name}] {text}")
    ctx = "\n".join(f"[{h[1]}] {h[2]}" for h in hits)
    prompt = (f"你是金融助手。仅基于资料回答，数字必须来自原文，附引用。\n"
              f"资料：\n{ctx}\n问题：{q}")
    resp = client.chat.completions.create(model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}])
    print("A:", resp.choices[0].message.content)

    print("\n" + "=" * 70)
    print("【多模态方式①：图像 caption → 文本 RAG】")
    # 用占位图 + 人工 caption 演示
    images = [
        ("chart.png", "2024年营收柱状图：Q1 25亿，Q2 28亿，Q3 32亿，Q4 35亿。"),
        ("map.png", "业务分布地图：华东40%，华南30%，华北20%，海外10%。"),
    ]
    q2 = "Q3营收多少？"
    hits2 = []
    for fname, cap in images:
        v = embed([cap])[0]
        qv = embed([q2])[0]
        hits2.append((float(v @ qv), fname, cap))
    hits2.sort(key=lambda x: -x[0])
    print("Q:", q2)
    for s, fname, cap in hits2:
        print(f"  score={s:.3f} [{fname}] {cap}")

    print("\n" + "=" * 70)
    print("【多模态方式③：图像索引 + 文本索引 + RRF 融合】")
    q3 = "营收"
    # 文本索引
    texts = [d[1] for d in finance_docs]
    Vt = embed(texts); qv = embed([q3])[0]
    vt = Vt @ qv
    t_rank = list(np.argsort(vt)[::-1][:3])
    # 图像索引
    img_caps = [c for _, c in images]
    Vi = embed(img_caps)
    vi = Vi @ qv
    i_rank = list(np.argsort(vi)[::-1][:3])
    # RRF
    rrf = {}
    for r, idx in enumerate(t_rank):
        rrf[("text", idx)] = rrf.get(("text", idx), 0) + 1 / (60 + r + 1)
    for r, idx in enumerate(i_rank):
        rrf[("image", idx)] = rrf.get(("image", idx), 0) + 1 / (60 + r + 1)
    print("Q:", q3)
    for (typ, idx), s in sorted(rrf.items(), key=lambda x: -x[1]):
        content = texts[idx] if typ == "text" else images[idx][1]
        print(f"  {typ}[{idx}] score={s:.3f}  {content[:40]}")

if __name__ == "__main__":
    main()
