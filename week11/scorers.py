# scorers.py
import re

def exact_match(pred: str, gold: str) -> float:
    """精确匹配：完全相同得 1.0，否则 0.0"""
    return 1.0 if pred.strip() == gold.strip() else 0.0

def keyword_match(pred: str, keywords: list) -> float:
    """关键词匹配：命中率 = 命中数 / 总关键词数"""
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in pred)
    return hits / len(keywords)

def llm_judge(pred: str, question: str, gold: str) -> dict:
    """LLM-as-Judge：返回 {score: 1-5, reason: str}"""
    # 模拟 LLM 评分（真实场景调用 LLM API）
    # 这里用简单规则模拟
    score = 3  # 默认 3 分
    if all(kw in pred for kw in ["检索", "生成"]):
        score = 5
    elif any(kw in pred for kw in ["检索", "生成"]):
        score = 4
    return {"score": score, "reason": f"包含 {sum(1 for kw in ['检索','生成'] if kw in pred)}/2 个核心概念"}

if __name__ == "__main__":
    pred = "RAG 是检索增强生成，通过检索外部知识来生成回答"
    gold = "RAG 是检索增强生成，结合检索和生成来回答问题"
    keywords = ["检索", "生成", "增强"]

    print(f"精确匹配: {exact_match(pred, gold)}")
    print(f"关键词匹配: {keyword_match(pred, keywords):.2f}")
    print(f"LLM 评分: {llm_judge(pred, '什么是 RAG？', gold)}")

