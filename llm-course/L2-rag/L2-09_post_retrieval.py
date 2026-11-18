# -*- coding: utf-8 -*-
"""L2-09 Post-Retrieval：LongContextReorder / RAG-Fusion / 上下文压缩"""
import os, hashlib, numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../../week1/.env")
client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4")

docs = [
    "公司年假：入职满1年享有5天年假，满3年10天，满5年15天。年假需提前3个工作日申请。",
    "报销流程：员工提交发票→直属经理审批→财务复核→3个工作日到账。单笔超过5000元需VP审批。",
    "加班补偿：工作日加班按1.5倍，周末2倍，法定节假日3倍。加班需提前申请并获批。",
    "体检福利：每年一次免费体检，入职满1年可带家属，家属体检半价。",
    "考勤：工作时间9:00-18:00，迟到15分钟内扣50元，超过30分钟按半天事假。",
    "离职：员工需提前30天书面通知，交接完成后发放工资与补偿。",
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

V = embed(docs)

def long_context_reorder(q, k=4):
    """检索 top-k，把最相关的移到两端（缓解 lost in the middle）。"""
    qv = embed([q])[0]
    scores = V @ qv
    top = list(np.argsort(scores)[::-1][:k])
    top_sorted = sorted(top, key=lambda i: -scores[i])
    # 相关度最高放开头，次高放结尾，交替
    reordered = []
    lo, hi = 0, len(top_sorted) - 1
    place_at_front = True
    while lo <= hi:
        if place_at_front:
            reordered.append(top_sorted[lo]); lo += 1
        else:
            reordered.append(top_sorted[hi]); hi -= 1
        place_at_front = not place_at_front
    return reordered

def rag_fusion(queries, k=60):
    """多查询 + RRF 融合。"""
    scores = np.zeros(len(docs))
    for q in queries:
        qv = embed([q])[0]
        sc = V @ qv
        rank = list(np.argsort(sc)[::-1][:5])
        for r, idx in enumerate(rank):
            scores[idx] += 1.0 / (k + r + 1)
    return list(np.argsort(scores)[::-1][:3])

def compress_chunk(chunk, q):
    """简单句子级压缩：保留与 query 共享字符较多的句子。"""
    sentences = [s for s in chunk.split("。") if s.strip()]
    qset = set(q)
    scored = []
    for s in sentences:
        sset = set(s)
        overlap = len(qset & sset) / max(len(qset), 1)
        scored.append((overlap, s))
    scored.sort(key=lambda x: -x[0])
    kept = [s for _, s in scored[:2]]
    return "。".join(kept) + "。"

def main():
    q = "年假怎么算？"
    print("Q:", q)

    print("\n--- LongContextReorder ---")
    order = long_context_reorder(q, k=4)
    for pos, idx in enumerate(order):
        print(f"pos{pos}: [{idx}] {docs[idx][:40]}")

    print("\n--- RAG-Fusion (多查询+RRF) ---")
    variants = [q, "年假天数", "带薪年假规定"]
    fused = rag_fusion(variants)
    for idx in fused:
        print(f"[{idx}] {docs[idx][:50]}")

    print("\n--- 上下文压缩 ---")
    chunk = docs[0]
    print("原文:", chunk)
    print("压缩:", compress_chunk(chunk, q))

if __name__ == "__main__":
    main()
