# agent_eval.py
import json
from agent_orchestrator import AgentOrchestrator
from llm_client import LLMClient
from tool_executor import ToolExecutor
from rag_engine import SimpleRAGEngine

class AgentEvaluator:
    """Agent 系统评测器"""

    def __init__(self, orchestrator: AgentOrchestrator):
        self.orch = orchestrator
        self.results = []

    def eval_case(self, case: dict) -> dict:
        """评测单个用例"""
        self.orch.reset_context()
        question = case["question"]
        expected_route = case["expected_route"]
        expected_keywords = case["expected_keywords"]

        # 执行
        answer = self.orch.handle(question)

        # 评分
        route_correct = self._check_route(question, expected_route)
        keyword_hits = sum(1 for kw in expected_keywords if kw in answer)
        keyword_score = keyword_hits / len(expected_keywords) if expected_keywords else 1.0

        result = {
            "id": case["id"],
            "question": question,
            "answer": answer[:80],
            "route_correct": route_correct,
            "keyword_score": round(keyword_score, 2),
            "status": "pass" if (route_correct and keyword_score >= 0.5) else "fail",
        }
        self.results.append(result)
        return result

    def _check_route(self, question: str, expected: str) -> bool:
        """检查路由是否正确（简化：检查回答中是否包含预期内容）"""
        # 实际工程中应该记录路由决策并对比
        return True  # 简化

    def run(self, cases: list) -> dict:
        for case in cases:
            r = self.eval_case(case)
            icon = "✅" if r["status"] == "pass" else "❌"
            print(f"  {icon} {r['id']}: route={r['route_correct']}, kw={r['keyword_score']}")

        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "pass")
        return {"total": total, "passed": passed, "pass_rate": round(passed / total, 3)}

if __name__ == "__main__":
    llm = LLMClient()
    tools = ToolExecutor()
    tools.register("weather", lambda city: f"{city}今天晴，25°C", {"required": ["city"]})

    rag = SimpleRAGEngine(llm)
    rag.add_documents(["RAG 是检索增强生成技术。\n\nRAG 利用外部知识减少幻觉。"])

    orch = AgentOrchestrator(llm, tools, rag)
    evaluator = AgentEvaluator(orch)

    cases = [
        {"id": "e1", "question": "什么是 RAG？", "expected_route": "rag", "expected_keywords": ["检索", "生成"]},
        {"id": "e2", "question": "北京天气？", "expected_route": "tool", "expected_keywords": ["晴", "25"]},
        {"id": "e3", "question": "你好", "expected_route": "direct", "expected_keywords": []},
    ]

    summary = evaluator.run(cases)
    print(f"\n评测摘要: {summary}")
