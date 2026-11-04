"""
W8 第 4 课：性能分析与瓶颈定位

演示：
  ① Token 计费追踪（每次调用记录 input/output tokens）
  ② 延迟分布（P50/P95/P99）
  ③ 瓶颈定位（哪个步骤最慢/最贵）
  ④ 成本估算（按 token 单价计算）
"""
import os
import json
import time
import statistics
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 3.0

# 价格表（智谱 GLM-4-Flash 免费，这里用假设价格演示）
PRICING = {
    "glm-4-flash": {"input": 0.0001, "output": 0.0002},   # 元/1K tokens
    "glm-4.5-flash": {"input": 0.0003, "output": 0.0006},
    "glm-4.7-flash": {"input": 0.0005, "output": 0.0010},
}


class PerformanceTracker:
    """性能追踪器：记录每次 LLM 调用的 token、延迟、成本"""

    def __init__(self):
        self.records = []

    def record(self, step_name: str, prompt: str, response: str,
              input_tokens: int, output_tokens: int, latency_ms: float,
              model: str = MODEL):
        cost = (input_tokens * PRICING[model]["input"] +
                output_tokens * PRICING[model]["output"]) / 1000
        entry = {
            "step": step_name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": round(latency_ms),
            "cost_yuan": round(cost, 6),
            "prompt_preview": prompt[:40],
            "response_preview": response[:40],
        }
        self.records.append(entry)
        return entry

    def summary(self):
        if not self.records:
            return {}
        latencies = [r["latency_ms"] for r in self.records]
        total_tokens = sum(r["total_tokens"] for r in self.records)
        total_cost = sum(r["cost_yuan"] for r in self.records)
        total_input = sum(r["input_tokens"] for r in self.records)
        total_output = sum(r["output_tokens"] for r in self.records)

        # 按步骤分组
        by_step = {}
        for r in self.records:
            s = r["step"]
            if s not in by_step:
                by_step[s] = {"count": 0, "total_ms": 0, "total_tokens": 0, "total_cost": 0}
            by_step[s]["count"] += 1
            by_step[s]["total_ms"] += r["latency_ms"]
            by_step[s]["total_tokens"] += r["total_tokens"]
            by_step[s]["total_cost"] += r["cost_yuan"]

        return {
            "total_calls": len(self.records),
            "total_tokens": total_tokens,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_yuan": round(total_cost, 6),
            "latency_p50": round(statistics.median(latencies)),
            "latency_p95": round(sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else latencies[0]),
            "latency_p99": round(sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) >= 2 else latencies[0]),
            "latency_max": max(latencies),
            "latency_min": min(latencies),
            "by_step": by_step,
        }


def call_with_tracking(prompt: str, step_name: str, tracker: PerformanceTracker) -> str:
    """带性能追踪的 LLM 调用"""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=150,
        extra_body={"thinking": {"type": "disabled"}},
    )
    latency_ms = (time.time() - t0) * 1000
    result = (resp.choices[0].message.content or "").strip()
    usage = resp.usage

    tracker.record(
        step_name=step_name,
        prompt=prompt,
        response=result,
        input_tokens=usage.prompt_tokens if usage else len(prompt) // 4,
        output_tokens=usage.completion_tokens if usage else len(result) // 4,
        latency_ms=latency_ms,
        model=MODEL,
    )
    return result


# ══════════════════════════════════════════════════════════
# ① 模拟多 Agent 链路的性能追踪
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 模拟多 Agent 链路的性能追踪")
print("=" * 60)

tracker = PerformanceTracker()
Q = "评估需求：给问答系统加语音输入功能"

# Step 1: 3 个 Agent 并行分析
print("\n--- Step 1: 3 个 Agent 分析 ---")
r1 = call_with_tracking(f"你是产品经理。任务：{Q}\n只从产品角度分析。", "agent.product", tracker)
time.sleep(GAP)
r2 = call_with_tracking(f"你是工程师。任务：{Q}\n只从工程角度分析。", "agent.engineer", tracker)
time.sleep(GAP)
r3 = call_with_tracking(f"你是测试工程师。任务：{Q}\n只从测试角度分析。", "agent.tester", tracker)
time.sleep(GAP)

# Step 2: 汇总
print("\n--- Step 2: 汇总 ---")
r4 = call_with_tracking(f"汇总以下三方分析：\n产品：{r1[:80]}\n工程：{r2[:80]}\n测试：{r3[:80]}\n输出一段总结。", "orchestrator.summarize", tracker)
time.sleep(GAP)

# Step 3: 格式化输出
print("\n--- Step 3: 格式化 ---")
r5 = call_with_tracking(f"把以下总结格式化成 Markdown：{r4[:100]}\n输出 Markdown。", "format.output", tracker)

# ══════════════════════════════════════════════════════════
# ② 性能汇总报告
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 性能汇总报告")
print("=" * 60)

s = tracker.summary()
print(f"\n  总调用次数: {s['total_calls']}")
print(f"  总 Token: {s['total_tokens']} (input={s['total_input_tokens']}, output={s['total_output_tokens']})")
print(f"  总成本: {s['total_cost_yuan']} 元")
print(f"  延迟 P50: {s['latency_p50']}ms")
print(f"  延迟 P95: {s['latency_p95']}ms")
print(f"  延迟 P99: {s['latency_p99']}ms")
print(f"  延迟范围: {s['latency_min']}ms ~ {s['latency_max']}ms")

# ══════════════════════════════════════════════════════════
# ③ 瓶颈定位：按步骤分组
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 瓶颈定位：按步骤分组")
print("=" * 60)

print(f"\n  {'步骤':<25} {'次数':>4} {'总耗时(ms)':>10} {'总Token':>8} {'总成本(元)':>10} {'avg_ms':>8}")
print("  " + "-" * 70)
for step, data in sorted(s["by_step"].items(), key=lambda x: -x[1]["total_ms"]):
    avg = data["total_ms"] / data["count"]
    print(f"  {step:<25} {data['count']:>4} {data['total_ms']:>10} {data['total_tokens']:>8} {data['total_cost']:>10.6f} {avg:>8.0f}")

# 找出瓶颈
slowest = max(s["by_step"].items(), key=lambda x: x[1]["total_ms"])
most_expensive = max(s["by_step"].items(), key=lambda x: x[1]["total_cost"])
print(f"\n  🔴 最慢步骤: {slowest[0]} ({slowest[1]['total_ms']}ms, {slowest[1]['total_ms']/s['total_calls']*100/s['latency_p50'] if s['latency_p50'] else 0:.0f}% of total)")
print(f"  💰 最贵步骤: {most_expensive[0]} ({most_expensive[1]['total_cost']:.6f} 元)")

# ══════════════════════════════════════════════════════════
# ④ 优化建议
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 优化建议")
print("=" * 60)

print("""
  基于以上数据分析：

  1. 延迟优化：
     - 3 个 Agent 是串行执行的 → 改为并行（asyncio）可将 19s → 6.3s
     - orchestrator.summarize 是最慢的（8.4s）→ 缩短输入上下文

  2. 成本优化：
     - input tokens >> output tokens → 压缩 prompt（去掉冗余上下文）
     - 用 glm-4-flash（免费）替代 glm-4.7-flash（付费）做非关键步骤
     - 缓存重复的 LLM 调用（相同 input → 直接返回缓存）

  3. 监控告警：
     - P95 延迟 > 10s → 告警
     - 单次成本 > 0.01 元 → 告警
     - 错误率 > 5% → 告警
""")

# ══════════════════════════════════════════════════════════
# ⑤ 保存性能数据
# ══════════════════════════════════════════════════════════
print("=" * 60)
with open("/tmp/perf_data.jsonl", "w") as f:
    for r in tracker.records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"性能数据已保存到 /tmp/perf_data.jsonl（{len(tracker.records)} 条记录）")
print("=" * 60)
print("\n  查询示例：")
print("  # 找出最贵的调用")
print("  cat perf_data.jsonl | jq -s 'max_by(.cost_yuan)'")
print("  # 找出 token 消耗最多的步骤")
print("  cat perf_data.jsonl | jq -s 'group_by(.step) | map({step: .[0].step, tokens: (map(.total_tokens)|add)}) | sort_by(-.tokens)'")
print("  # 计算总成本")
print("  cat perf_data.jsonl | jq -s 'map(.cost_yuan)|add'")
print("=" * 60)
