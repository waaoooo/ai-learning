# -*- coding: utf-8 -*-
"""L2-12 三平台工作流模拟对比（非真实部署）
模拟：文档解析 → 分块 → 索引 → 混合检索 → rerank → 生成（带引用）
"""
import hashlib, numpy as np

DOC = ("公司年假制度：入职满1年享有5天年假，满3年10天，满5年15天。"
       "报销制度：员工提交发票→经理审批→财务复核→3个工作日到账。"
       "加班制度：工作日1.5倍，周末2倍，法定节假日3倍。")

def local_embed(texts, dim=128):
    out = []
    for t in texts:
        v = np.zeros(dim, dtype="float32")
        for i in range(len(t)):
            for n in (1, 2):
                if i + n <= len(t):
                    h = int(hashlib.md5(t[i:i + n].encode()).hexdigest(), 16)
                    v[h % dim] += 1.0
        nrm = np.linalg.norm(v)
        if nrm > 0: v /= nrm
        out.append(v)
    return np.array(out, dtype="float32")

def bm25_scores(corpus, q):
    qset = set(q)
    out = []
    for d in corpus:
        dset = set(d)
        inter = len(qset & dset)
        out.append(inter / ((len(qset) * len(dset)) ** 0.5 + 1e-9))
    return np.array(out)

def run_platform(name, parse_note, chunk_fn, use_hybrid, use_rerank):
    print("=" * 68)
    print(f"平台: {name}")
    print(f"  文档解析: {parse_note}")
    chunks = chunk_fn(DOC)
    V = local_embed(chunks)
    q = "年假怎么算？"
    qv = local_embed([q])[0]
    vec = V @ qv
    if use_hybrid:
        kw = bm25_scores(chunks, q)
        vn = (vec - vec.min()) / (vec.max() - vec.min() + 1e-9)
        kn = (kw - kw.min()) / (kw.max() - kw.min() + 1e-9)
        fused = 0.6 * vn + 0.4 * kn
    else:
        fused = vec
    order = list(np.argsort(fused)[::-1])
    if use_rerank:
        order.sort(key=lambda i: -len(set(chunks[i]) & set(q)))
    top = order[:2]
    print(f"  分块数: {len(chunks)}  混合检索: {use_hybrid}  rerank: {use_rerank}")
    print(f"  查询: {q}")
    for i in top:
        print(f"    [{i}] score={fused[i]:.3f}  {chunks[i][:40]}")

def chunk_fixed(text, size=60, overlap=10):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size]); i += size - overlap
    return out

def chunk_sentence(text):
    return [s for s in text.replace("。", "。\n").split("\n") if s.strip()]

def main():
    run_platform("RAGFlow", "深度解析(表格/多栏/扫描件)",
                 chunk_sentence, use_hybrid=True, use_rerank=True)
    run_platform("FastGPT", "常规解析",
                 lambda t: chunk_fixed(t, 60, 10), use_hybrid=True, use_rerank=False)
    run_platform("Dify", "常规解析",
                 lambda t: chunk_fixed(t, 50, 5), use_hybrid=False, use_rerank=True)

if __name__ == "__main__":
    main()
