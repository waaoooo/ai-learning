# eval_report.py
import json
import os

class EvalReport:
    def __init__(self, results: list, summary: dict):
        self.results = results
        self.summary = summary

    def render(self) -> str:
        lines = []
        lines.append("=" * 50)
        lines.append("评测报告")
        lines.append("=" * 50)
        lines.append(f"\n总题数: {self.summary['total']}")
        lines.append(f"通过: {self.summary['passed']} ({self.summary['pass_rate']*100:.1f}%)")
        lines.append(f"平均关键词得分: {self.summary['avg_keyword_score']:.3f}")
        lines.append(f"平均 LLM 评分: {self.summary['avg_judge_score']:.3f}")

        # 按难度分组
        by_diff = {}
        for r in self.results:
            diff = r.get("difficulty", "unknown")
            by_diff.setdefault(diff, []).append(r)

        lines.append("\n--- 按难度 ---")
        for diff, cases in by_diff.items():
            passed = sum(1 for c in cases if c["status"] == "pass")
            lines.append(f"  {diff}: {passed}/{len(cases)} 通过")

        lines.append("\n--- 详细结果 ---")
        for r in self.results:
            icon = "✅" if r["status"] == "pass" else "❌"
            lines.append(f"  {icon} {r['id']}: kw={r.get('keyword_score',0):.2f} judge={r.get('judge_score',0)}")

        return "\n".join(lines)

    def save_baseline(self, path: str):
        """保存基线分数"""
        baseline = {
            "pass_rate": self.summary["pass_rate"],
            "avg_keyword_score": self.summary["avg_keyword_score"],
            "avg_judge_score": self.summary["avg_judge_score"],
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"  基线已保存: {path}")

    def check_regression(self, baseline_path: str) -> dict:
        """回归检查：对比当前结果与基线"""
        if not os.path.exists(baseline_path):
            return {"status": "no_baseline", "message": "无基线，跳过回归检查"}

        with open(baseline_path) as f:
            baseline = json.load(f)

        current_acc = self.summary["pass_rate"]
        baseline_acc = baseline["pass_rate"]
        drop = baseline_acc - current_acc

        if drop > 0.02:
            return {
                "status": "REGRESSION",
                "message": f"准确率下降 {drop*100:.1f}% (基线 {baseline_acc:.3f} → 当前 {current_acc:.3f})，阻断发布",
            }
        return {
            "status": "OK",
            "message": f"准确率变化 {drop*100:+.1f}%，在允许范围内",
        }


if __name__ == "__main__":
    # 模拟评测结果
    results = [
        {"id": "q1", "status": "pass", "keyword_score": 1.0, "judge_score": 5, "difficulty": "simple"},
        {"id": "q2", "status": "pass", "keyword_score": 1.0, "judge_score": 4, "difficulty": "medium"},
        {"id": "q3", "status": "pass", "keyword_score": 0.8, "judge_score": 4, "difficulty": "complex"},
    ]
    summary = {"total": 3, "passed": 3, "pass_rate": 1.0, "avg_keyword_score": 0.933, "avg_judge_score": 4.333}

    report = EvalReport(results, summary)
    print(report.render())

    # 保存基线
    report.save_baseline("/tmp/baseline.json")

    # 模拟回归：准确率下降 33%
    print("\n--- 模拟回归检查 ---")
    bad_results = [
        {"id": "q1", "status": "pass", "keyword_score": 1.0, "judge_score": 5, "difficulty": "simple"},
        {"id": "q2", "status": "fail", "keyword_score": 0.25, "judge_score": 2, "difficulty": "medium"},
        {"id": "q3", "status": "fail", "keyword_score": 0.4, "judge_score": 2, "difficulty": "complex"},
    ]
    bad_summary = {"total": 3, "passed": 1, "pass_rate": 0.333, "avg_keyword_score": 0.55, "avg_judge_score": 3.0}
    bad_report = EvalReport(bad_results, bad_summary)
    reg = bad_report.check_regression("/tmp/baseline.json")
    print(f"  回归状态: {reg['status']}")
    print(f"  {reg['message']}")
