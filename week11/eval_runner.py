# eval_runner.py
import time
from eval_dataset import EVAL_DATASET
from scorers import keyword_match, llm_judge

class EvalRunner:
    def __init__(self, agent_fn):
        self.agent_fn = agent_fn  # agent_fn(question) -> str
        self.results = []

    def run(self) -> dict:
        for case in EVAL_DATASET:
            start = time.time()
            try:
                pred = self.agent_fn(case["question"])
                latency = time.time() - start

                kw_score = keyword_match(pred, case["expected_keywords"])
                judge_result = llm_judge(pred, case["question"], case["gold_answer"])

                result = {
                    "id": case["id"],
                    "question": case["question"],
                    "prediction": pred,
                    "keyword_score": kw_score,
                    "judge_score": judge_result["score"],
                    "latency": round(latency, 3),
                    "status": "pass" if kw_score >= 0.5 else "fail",
                }
            except Exception as e:
                result = {
                    "id": case["id"],
                    "question": case["question"],
                    "prediction": None,
                    "error": str(e),
                    "status": "error",
                }
            self.results.append(result)
            print(f"  {result['id']}: {result['status']} (kw={result.get('keyword_score', 0):.2f})")

        return self.summary()

    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "pass")
        avg_kw = sum(r.get("keyword_score", 0) for r in self.results) / total
        avg_judge = sum(r.get("judge_score", 0) for r in self.results) / total
        avg_latency = sum(r.get("latency", 0) for r in self.results) / total
        return {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 3),
            "avg_keyword_score": round(avg_kw, 3),
            "avg_judge_score": round(avg_judge, 3),
            "avg_latency": round(avg_latency, 3),
        }


# === 模拟 Agent ===
def mock_agent(question: str) -> str:
    """模拟 Agent 回答（真实场景替换为实际 Agent）"""
    responses = {
        "什么是 RAG？": "RAG 是检索增强生成，通过检索外部知识来生成回答",
        "RAG 和微调有什么区别？": "RAG 通过检索外部知识回答，微调改变模型参数",
        "设计一个企业级 RAG 系统需要考虑哪些因素？": "需要考虑检索精度、延迟、成本、安全",
    }
    return responses.get(question, "不知道")

if __name__ == "__main__":
    runner = EvalRunner(mock_agent)
    summary = runner.run()
    print(f"\n评测摘要: {summary}")
