"""
W11 第 2 课：自动化评测框架

演示：
  ① 评测集设计（JSON 格式、难度分层）
  ② 评分器（精确匹配/关键词/LLM-as-Judge）
  ③ 评测调度器（批量运行、并发控制）
  ④ 评测报告（准确率、成本、延迟、回归对比）
  ⑤ 回归测试（baseline 对比、自动告警）
"""
import os
import json
import time
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 1.5


def call(messages, temperature=0.0, system=None, max_tokens=100) -> dict:
    start = time.time()
    try:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        resp = client.chat.completions.create(
            model=MODEL, messages=msgs, temperature=temperature,
            max_tokens=max_tokens, extra_body={"thinking": {"type": "disabled"}},
        )
        return {"content": (resp.choices[0].message.content or "").strip(),
                "tokens": resp.usage.total_tokens if resp.usage else 0,
                "latency": time.time() - start, "error": None}
    except Exception as e:
        return {"content": f"[ERROR] {e}", "tokens": 0, "latency": time.time() - start, "error": str(e)}


def clean_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = __import__("re").search(r"\{.*\}", text, __import__("re").DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


# ══════════════════════════════════════════════════════════
# ① 评测集设计
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 评测集设计")
print("=" * 60)

# 评测集：JSON 格式，包含难度分层
eval_dataset = [
    # simple: 直接事实查询
    {"id": "E001", "difficulty": "simple", "question": "客服系统支持哪些功能？",
     "expected_keywords": ["多轮对话", "知识库", "检索"], "gold_answer": "支持多轮对话、知识库检索、自动回复"},
    {"id": "E002", "difficulty": "simple", "question": "产品价格是多少？",
     "expected_keywords": ["999", "元"], "gold_answer": "客服系统 999元/月"},
    # medium: 需要推理
    {"id": "E003", "difficulty": "medium", "question": "产品 A 和产品 B 哪个更适合中小企业？",
     "expected_keywords": ["中小企业", "性价比", "推荐"], "gold_answer": "产品 A 性价比更高，推荐中小企业"},
    {"id": "E004", "difficulty": "medium", "question": "对比三个产品的优缺点",
     "expected_keywords": ["客服", "数据分析", "项目管理", "优缺点"], "gold_answer": "客服系统易用但功能有限；数据分析功能强但学习成本高；项目管理协作好但价格贵"},
    # complex: 需要生成
    {"id": "E005", "difficulty": "complex", "question": "帮我设计一个基于 RAG 的客服系统架构",
     "expected_keywords": ["RAG", "检索", "向量", "LLM", "架构"], "gold_answer": "包含文档切片、向量存储、检索器、LLM 生成、置信度门控"},
]

print(f"  评测集: {len(eval_dataset)} 个用例")
for d in ["simple", "medium", "complex"]:
    count = sum(1 for e in eval_dataset if e["difficulty"] == d)
    print(f"    {d}: {count} 个")

# 评测集统计
print(f"\n  评测集字段: id, difficulty, question, expected_keywords, gold_answer")
print(f"  难度分布: simple={sum(1 for e in eval_dataset if e['difficulty']=='simple')}, "
      f"medium={sum(1 for e in eval_dataset if e['difficulty']=='medium')}, "
      f"complex={sum(1 for e in eval_dataset if e['difficulty']=='complex')}")


# ══════════════════════════════════════════════════════════
# ② 评分器
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 评分器 (Scorer)")
print("=" * 60)

class Scorer:
    """多维度评分器"""

    @staticmethod
    def exact_match(answer: str, gold: str) -> float:
        """精确匹配"""
        return 1.0 if answer.strip() == gold.strip() else 0.0

    @staticmethod
    def keyword_match(answer: str, keywords: list[str]) -> float:
        """关键词匹配：命中关键词比例"""
        if not keywords:
            return 0.0
        hits = sum(1 for kw in keywords if kw in answer)
        return hits / len(keywords)

    @staticmethod
    def llm_judge(question: str, answer: str, gold: str) -> dict:
        """LLM-as-Judge 评分"""
        prompt = f"""评估以下回答的质量（1-5分）。
问题：{question}
标准答案：{gold}
实际回答：{answer}

评分维度：
1. 正确性：是否准确回答了问题
2. 完整性：是否覆盖了关键信息
3. 简洁性：是否简洁明了

输出 JSON：{{"score": 1-5, "reason": "理由"}}"""
        r = call([{"role": "user", "content": prompt}], system="你是严格的质量评估员，只输出 JSON。")
        result = clean_json(r["content"])
        return {"score": result.get("score", 0), "reason": result.get("reason", ""), "tokens": r["tokens"]}

    @staticmethod
    def combined_score(answer: str, gold: str, keywords: list[str], question: str) -> dict:
        """综合评分"""
        em = Scorer.exact_match(answer, gold)
        km = Scorer.keyword_match(answer, keywords)
        judge = Scorer.llm_judge(question, answer, gold)
        # 综合分数：关键词 40% + LLM Judge 60%
        combined = km * 0.4 + (judge["score"] / 5) * 0.6
        return {
            "exact_match": em,
            "keyword_match": km,
            "llm_score": judge["score"],
            "llm_reason": judge["reason"],
            "combined": round(combined, 3),
            "tokens": judge["tokens"],
        }


# 演示评分器
print("  评分器演示:")
test_answer = "客服系统支持多轮对话、知识库检索和自动回复功能"
sc = Scorer.combined_score(test_answer, "支持多轮对话、知识库检索、自动回复",
                           ["多轮对话", "知识库", "检索"], "客服系统支持哪些功能？")
print(f"    回答: {test_answer[:40]}")
print(f"    exact_match: {sc['exact_match']}")
print(f"    keyword_match: {sc['keyword_match']}")
print(f"    llm_score: {sc['llm_score']}/5 ({sc['llm_reason']})")
print(f"    combined: {sc['combined']}")


# ══════════════════════════════════════════════════════════
# ③ 评测调度器
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 评测调度器 (Eval Runner)")
print("=" * 60)

class EvalRunner:
    """评测调度器：批量运行、并发控制、结果收集"""

    def __init__(self, model_system_prompt: str = None):
        self.system_prompt = model_system_prompt or "你是客服助手，回答产品相关问题。"
        self.results = []

    def run_single(self, eval_case: dict) -> dict:
        """运行单个评测用例"""
        start = time.time()
        r = call([{"role": "user", "content": eval_case["question"]}],
                 system=self.system_prompt, max_tokens=150)
        latency = time.time() - start

        score = Scorer.combined_score(
            r["content"], eval_case["gold_answer"],
            eval_case["expected_keywords"], eval_case["question"]
        )

        result = {
            "id": eval_case["id"],
            "difficulty": eval_case["difficulty"],
            "question": eval_case["question"],
            "answer": r["content"],
            "gold": eval_case["gold_answer"],
            "score": score,
            "latency": round(latency, 2),
            "tokens": r["tokens"],
            "error": r["error"],
        }
        self.results.append(result)
        return result

    def run_batch(self, dataset: list[dict], verbose: bool = True) -> list[dict]:
        """批量运行评测"""
        for i, case in enumerate(dataset):
            if verbose:
                print(f"  [{i+1}/{len(dataset)}] {case['id']} ({case['difficulty']})...", end=" ", flush=True)
            result = self.run_single(case)
            if verbose:
                print(f"score={result['score']['combined']:.2f}, latency={result['latency']}s")
            time.sleep(GAP)
        return self.results

    def summary(self) -> dict:
        """生成评测摘要"""
        if not self.results:
            return {}
        scores = [r["score"]["combined"] for r in self.results]
        latencies = [r["latency"] for r in self.results]
        tokens = [r["tokens"] for r in self.results]
        errors = [r for r in self.results if r["error"]]

        by_diff = {}
        for d in ["simple", "medium", "complex"]:
            subset = [r for r in self.results if r["difficulty"] == d]
            if subset:
                by_diff[d] = {
                    "count": len(subset),
                    "avg_score": round(sum(r["score"]["combined"] for r in subset) / len(subset), 3),
                    "avg_latency": round(sum(r["latency"] for r in subset) / len(subset), 2),
                }

        return {
            "total": len(self.results),
            "avg_score": round(sum(scores) / len(scores), 3),
            "min_score": round(min(scores), 3),
            "max_score": round(max(scores), 3),
            "avg_latency": round(sum(latencies) / len(latencies), 2),
            "p95_latency": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "total_tokens": sum(tokens),
            "avg_tokens": round(sum(tokens) / len(tokens), 0),
            "error_count": len(errors),
            "by_difficulty": by_diff,
        }


# 运行评测
runner = EvalRunner()
results = runner.run_batch(eval_dataset)
summary = runner.summary()

print(f"\n  评测摘要:")
print(f"    总用例: {summary['total']}")
print(f"    平均分: {summary['avg_score']}")
print(f"    最低分: {summary['min_score']}, 最高分: {summary['max_score']}")
print(f"    平均延迟: {summary['avg_latency']}s, P95: {summary['p95_latency']}s")
print(f"    总 Token: {summary['total_tokens']}, 平均: {summary['avg_tokens']}")
print(f"    错误数: {summary['error_count']}")
print(f"\n  按难度:")
for d, info in summary["by_difficulty"].items():
    print(f"    {d}: {info['count']} 用例, 平均分 {info['avg_score']}, 平均延迟 {info['avg_latency']}s")


# ══════════════════════════════════════════════════════════
# ④ 评测报告
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 评测报告")
print("=" * 60)

class EvalReport:
    """评测报告生成器"""

    @staticmethod
    def generate(results: list[dict], summary: dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("评测报告")
        lines.append("=" * 60)
        lines.append(f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"模型: {MODEL}")
        lines.append(f"\n--- 摘要 ---")
        lines.append(f"总用例: {summary['total']}")
        lines.append(f"平均分: {summary['avg_score']}/1.0")
        lines.append(f"最低分: {summary['min_score']}, 最高分: {summary['max_score']}")
        lines.append(f"平均延迟: {summary['avg_latency']}s")
        lines.append(f"P95 延迟: {summary['p95_latency']}s")
        lines.append(f"总 Token: {summary['total_tokens']}")
        lines.append(f"错误数: {summary['error_count']}")

        lines.append(f"\n--- 按难度 ---")
        for d, info in summary["by_difficulty"].items():
            lines.append(f"  {d}: {info['count']} 用例, 平均分 {info['avg_score']}, 延迟 {info['avg_latency']}s")

        lines.append(f"\n--- 详细结果 ---")
        for r in results:
            status = "✅" if r["score"]["combined"] >= 0.6 else "❌"
            lines.append(f"  {status} {r['id']} ({r['difficulty']}) score={r['score']['combined']:.2f}")
            lines.append(f"      Q: {r['question'][:40]}")
            lines.append(f"      A: {r['answer'][:60]}")
            lines.append(f"      Gold: {r['gold'][:40]}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

report = EvalReport.generate(results, summary)
print(report)


# ══════════════════════════════════════════════════════════
# ⑤ 回归测试
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑤ 回归测试 (Regression Testing)")
print("=" * 60)

class RegressionTester:
    """回归测试：对比当前结果与 baseline"""

    def __init__(self, baseline_path: str = None):
        self.baseline = {}
        if baseline_path and os.path.exists(baseline_path):
            with open(baseline_path) as f:
                self.baseline = json.load(f)

    def compare(self, current_summary: dict) -> dict:
        """对比当前结果与 baseline"""
        if not self.baseline:
            print("  无 baseline，保存当前结果为 baseline")
            return {"status": "no_baseline", "saved": True}

        comparisons = {}
        for metric in ["avg_score", "avg_latency", "total_tokens"]:
            base_val = self.baseline.get(metric, 0)
            curr_val = current_summary.get(metric, 0)
            if base_val > 0:
                change = (curr_val - base_val) / base_val * 100
            else:
                change = 0
            comparisons[metric] = {
                "baseline": base_val,
                "current": curr_val,
                "change_pct": round(change, 1),
            }

        # 判断是否回归
        score_change = comparisons["avg_score"]["change_pct"]
        latency_change = comparisons["avg_latency"]["change_pct"]

        if score_change < -2:  # 准确率下降超过 2%
            status = "REGRESSION"
            message = f"⚠️ 准确率下降 {abs(score_change):.1f}%，可能存在回归！"
        elif latency_change > 20:  # 延迟增加超过 20%
            status = "PERFORMANCE_DEGRADATION"
            message = f"⚠️ 延迟增加 {latency_change:.1f}%，性能退化！"
        else:
            status = "OK"
            message = "✅ 无回归，指标正常"

        return {"status": status, "message": message, "comparisons": comparisons}

    def save_baseline(self, summary: dict, path: str):
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Baseline 已保存: {path}")


# 模拟 baseline（假设上次评测结果）
baseline = {
    "avg_score": 0.72,
    "avg_latency": 3.5,
    "total_tokens": 1200,
}
baseline_path = os.path.expanduser("~/hermes/ai-learning/week11/baseline.json")
with open(baseline_path, "w") as f:
    json.dump(baseline, f)

tester = RegressionTester(baseline_path)
comparison = tester.compare(summary)

print(f"  回归测试结果: {comparison['status']}")
print(f"  {comparison['message']}")
print(f"\n  指标对比:")
for metric, info in comparison["comparisons"].items():
    arrow = "↑" if info["change_pct"] > 0 else "↓" if info["change_pct"] < 0 else "→"
    print(f"    {metric}: {info['baseline']} → {info['current']} ({arrow}{abs(info['change_pct']):.1f}%)")

# 保存当前结果为新 baseline
tester.save_baseline(summary, baseline_path)

print("\n" + "=" * 60)
print("W11 第 2 课总结")
print("=" * 60)
print("""
  ① 评测集：JSON 格式，难度分层 (simple/medium/complex)
  ② 评分器：精确匹配 + 关键词匹配 + LLM-as-Judge 综合
  ③ 评测调度：批量运行、并发控制、结果收集
  ④ 评测报告：准确率、成本、延迟、按难度分组
  ⑤ 回归测试：baseline 对比、自动告警

  评测是 Agent 系统的"体检" —— 定期跑，持续监控
  回归测试是"安全网" —— 代码改动后自动检测退化
""")
print("=" * 60)
print("\n  问你一个问题（引向 W11 第 3 课）：")
print("  CI/CD 流水线应该包含哪些阶段？")
print("  （Lint → Test → Build → Deploy → Verify）")
print("=" * 60)
