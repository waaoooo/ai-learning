# -*- coding: utf-8 -*-
"""L2-11 RAG 评估：手工实现 5 个指标（无 ragas 依赖）"""
import os, hashlib, math, numpy as np

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

def sim(a, b):
    return float(np.dot(a, b))

def context_relevancy(query, contexts, threshold=0.15):
    """相关 chunk 比例"""
    qv = _local_embed([query])[0]
    rel = sum(1 for c in contexts if sim(qv, _local_embed([c])[0]) >= threshold)
    return rel / max(len(contexts), 1)

def context_precision(query, contexts, threshold=0.15):
    """相关 chunk 的排序质量（越靠前权重越高）"""
    qv = _local_embed([query])[0]
    scores = [sim(qv, _local_embed([c])[0]) for c in contexts]
    ranked = [s >= threshold for s in scores]
    if not any(ranked):
        return 0.0
    # 类似 MRR 的加权：相关项按位置加权
    precisions = []
    for i, r in enumerate(ranked):
        if r:
            precisions.append(sum(ranked[:i + 1]) / (i + 1))
    return sum(precisions) / len(precisions)

def context_recall(retrieved, ground_truth, threshold=0.15):
    """ground truth 信息被召回的比例"""
    gt = _local_embed([ground_truth])[0]
    hit = sum(1 for c in retrieved if sim(gt, _local_embed([c])[0]) >= threshold)
    return hit / max(len(retrieved), 1)

def answer_relevancy(answer, query, threshold=0.15):
    """答案与查询的相关性"""
    av = _local_embed([answer])[0]
    qv = _local_embed([query])[0]
    return 1.0 if sim(av, qv) >= threshold else 0.0

def faithfulness(answer, contexts, threshold=0.12):
    """答案中句子能在上下文中找到依据的比例（简化）"""
    ctx = "。".join(contexts)
    sentences = [s for s in answer.split("。") if s.strip()]
    if not sentences:
        return 0.0
    cv = _local_embed([ctx])[0]
    supported = 0
    for s in sentences:
        sv = _local_embed([s])[0]
        if sim(sv, cv) >= threshold:
            supported += 1
    return supported / len(sentences)

def main():
    samples = [
        {
            "question": "年假怎么算？",
            "answer": "入职满1年享有5天年假，满3年10天，满5年15天。",
            "contexts": [
                "公司年假：入职满1年享有5天年假，满3年10天，满5年15天。",
                "报销流程：员工提交发票→经理审批→财务复核。",
            ],
            "ground_truth": "年假按入职年限计算。",
        },
        {
            "question": "公司有股票期权吗？",
            "answer": "资料中未提及股票期权。",
            "contexts": [
                "报销流程：员工提交发票→经理审批→财务复核。",
                "考勤：工作时间9:00-18:00。",
            ],
            "ground_truth": "无相关信息",
        },
    ]

    print("RAG 评估指标（手工实现）\n" + "=" * 60)
    for i, s in enumerate(samples, 1):
        print(f"\n样本{i}: {s['question']}")
        cr = context_relevancy(s["question"], s["contexts"])
        cp = context_precision(s["question"], s["contexts"])
        crec = context_recall(s["contexts"], s["ground_truth"])
        ar = answer_relevancy(s["answer"], s["question"])
        f = faithfulness(s["answer"], s["contexts"])
        print(f"  Context Relevancy : {cr:.3f}")
        print(f"  Context Precision : {cp:.3f}")
        print(f"  Context Recall    : {crec:.3f}")
        print(f"  Answer Relevancy  : {ar:.3f}")
        print(f"  Faithfulness      : {f:.3f}")

if __name__ == "__main__":
    main()
