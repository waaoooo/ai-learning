# -*- coding: utf-8 -*-
"""L2-08 Pre-Retrieval：父子索引 + Multi-Query"""
import os, hashlib, numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../../week1/.env")
client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4")

# 父子结构：parent = 完整制度；child = 细分点
parent_docs = [
    ("年假与病假制度",
     "年假：入职满1年5天，满3年10天，满5年15天，需提前3天申请。"
     "病假：需提供医院证明，每年不超过30天，病假期间工资按80%发放。"
     "两者区别：年假需提前申请，病假需医疗证明；年假带薪100%，病假80%。"),
    ("报销与差旅制度",
     "报销：员工提交发票→经理审批→财务复核→3个工作日到账，超5000元需VP审批。"
     "差旅：国内出差每天补贴200元，住宿上限500元/晚，需保存发票。"),
]
child_docs = [
    "年假：入职满1年5天，满3年10天，满5年15天。",
    "病假：需医院证明，每年不超过30天，工资按80%发放。",
    "年假与病假区别：年假需提前申请、带薪100%；病假需证明、80%。",
    "报销：提交发票→经理审批→财务复核→3工作日，超5000需VP审批。",
    "差旅：国内每天补贴200元，住宿上限500元/晚。",
]
child_to_parent = [0, 0, 0, 1, 1]

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

def main():
    cv = embed(child_docs)
    cv_norm = cv / np.clip(np.linalg.norm(cv, axis=1, keepdims=True), 1e-12, None)

    q = "年假和病假有什么区别？"
    qv = embed([q])[0]
    scores = cv_norm @ qv
    top = np.argsort(scores)[::-1][:2]

    print("Q:", q)
    print("\n--- 父子索引：child 检索 ---")
    for i in top:
        print(f"child[{i}] score={scores[i]:.3f}  {child_docs[i]}")
    parents = sorted(set(child_to_parent[i] for i in top))
    print("\n--- 命中后返回 parent ---")
    for p in parents:
        print(f"parent[{p}] {parent_docs[p][0]}: {parent_docs[p][1][:80]}...")

    # Multi-Query
    print("\n--- Multi-Query ---")
    variants = [q, "病假工资怎么算？", "病假需要什么证明？"]
    all_hits = {}
    for v in variants:
        vv = embed([v])[0]
        sc = cv_norm @ vv
        idx = int(np.argmax(sc))
        all_hits[idx] = max(all_hits.get(idx, -1), float(sc[idx]))
        print(f"变体[{v}] -> child[{idx}] score={sc[idx]:.3f} {child_docs[idx][:30]}")
    print("\n合并去重后:")
    for idx, s in sorted(all_hits.items(), key=lambda x: -x[1]):
        print(f"  child[{idx}] score={s:.3f}  {child_docs[idx]}")

if __name__ == "__main__":
    main()
