# rag_engine.py
import math
from collections import Counter

class SimpleRAGEngine:
    """简化 RAG 引擎：TF-IDF 检索 + LLM 生成"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.chunks = []
        self.vectors = []

    def add_documents(self, docs: list[str]):
        """添加文档并切片"""
        for doc in docs:
            # 按段落切片
            for para in doc.split("\n\n"):
                if para.strip():
                    self.chunks.append(para.strip())
        # 向量化
        self.vectors = [self._tfidf(c) for c in self.chunks]

    def _tfidf(self, text: str) -> dict:
        """字符级 TF-IDF"""
        chars = list(text)
        tf = Counter(chars)
        total = len(chars)
        return {c: count / total for c, count in tf.items()}

    def _cosine(self, v1: dict, v2: dict) -> float:
        """余弦相似度"""
        common = set(v1) & set(v2)
        if not common:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common)
        n1 = math.sqrt(sum(v * v for v in v1.values()))
        n2 = math.sqrt(sum(v * v for v in v2.values()))
        return dot / (n1 * n2) if n1 and n2 else 0.0

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """检索相关 chunks"""
        q_vec = self._tfidf(query)
        scored = [(self._cosine(q_vec, v), chunk)
                  for v, chunk in zip(self.vectors, self.chunks)]
        scored.sort(reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def answer(self, question: str, top_k: int = 3) -> str:
        """RAG 回答"""
        chunks = self.retrieve(question, top_k)
        context = "\n".join(chunks)
        messages = [
            {"role": "system", "content": f"根据以下知识回答问题：\n{context}"},
            {"role": "user", "content": question},
        ]
        return self.llm.chat(messages)

if __name__ == "__main__":
    from llm_client import LLMClient
    llm = LLMClient()
    rag = SimpleRAGEngine(llm)

    # 添加知识库
    rag.add_documents([
        "RAG 是检索增强生成技术，结合信息检索和文本生成。\n\nRAG 的优势是能利用外部知识，减少幻觉。",
        "微调是通过训练数据调整模型参数。\n\n微调需要大量数据和计算资源。",
    ])

    # 查询
    question = "RAG 和微调有什么区别？"
    chunks = rag.retrieve(question)
    print(f"检索到 {len(chunks)} 个 chunks:")
    for c in chunks:
        print(f"  - {c[:50]}...")

    answer = rag.answer(question)
    print(f"\n回答: {answer}")
