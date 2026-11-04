# eval_dataset.py
import json

EVAL_DATASET = [
    {
        "id": "q1",
        "question": "什么是 RAG？",
        "gold_answer": "RAG 是检索增强生成，结合检索和生成来回答问题",
        "expected_keywords": ["检索", "生成", "增强"],
        "difficulty": "simple",
        "tags": ["基础概念"],
    },
    {
        "id": "q2",
        "question": "RAG 和微调有什么区别？",
        "gold_answer": "RAG 通过检索外部知识回答，微调通过训练改变模型参数",
        "expected_keywords": ["检索", "微调", "参数", "知识"],
        "difficulty": "medium",
        "tags": ["对比分析"],
    },
    {
        "id": "q3",
        "question": "设计一个企业级 RAG 系统需要考虑哪些因素？",
        "gold_answer": "需要考虑检索精度、响应延迟、成本、安全、可扩展性等",
        "expected_keywords": ["检索精度", "延迟", "成本", "安全", "扩展"],
        "difficulty": "complex",
        "tags": ["系统设计"],
    },
]

if __name__ == "__main__":
    print(f"评测集大小: {len(EVAL_DATASET)}")
    for case in EVAL_DATASET:
        print(f"  [{case['difficulty']}] {case['id']}: {case['question']}")
        print(f"    关键词: {case['expected_keywords']}")

