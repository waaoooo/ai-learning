"""
W8 第 1 课：为什么需要可观测性

演示：一个 3 步 Agent 链路，中间某步静默失败（返回空/错误），
如果没有日志，你根本不知道是哪一步、哪次调用出了问题。

对比：
  ① 黑盒模式：直接跑，出错只看到最终答案错，不知道错在哪
  ② 加日志后：能看到每步的输入/输出/耗时/状态
"""
import os
import time
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 3.0


def call(prompt, temperature=0.0) -> str:
    last_err: Exception = RuntimeError("call failed")
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=150,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err


# ══════════════════════════════════════════════════════════
# 模拟一个 3 步 Agent 链路
# ══════════════════════════════════════════════════════════
def step1_extract(text):
    """步骤1：从文本提取关键信息"""
    return call(f"从以下文本提取年假天数：{text}\n只输出数字。")

def step2_calculate(days_str):
    """步骤2：计算剩余天数"""
    # 故意埋一个 bug：如果输入不是纯数字，返回"ERROR"
    try:
        days = int(days_str.strip())
        return str(days - 3)   # 假设用了3天
    except ValueError:
        return "ERROR: invalid input"

def step3_format(result):
    """步骤3：格式化输出"""
    return call(f"把以下结果格式化成友好回复：{result}\n输出一句话。")


# ══════════════════════════════════════════════════════════
# ① 黑盒模式：直接跑，出错你不知道在哪
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 黑盒模式：直接跑")
print("=" * 60)

text = "公司规定年假15天，病假30天。"
print(f"\n  输入: {text}")

r1 = step1_extract(text)
print(f"  最终答案: {r1}")   # 你只看到最终输出

# 故意给一个会失败的输入
text2 = "年假和病假各有多少天？"
print(f"\n  输入: {text2}")
r2 = step1_extract(text2)
print(f"  最终答案: {r2}")   # 你只看到最终输出，不知道中间哪步错了

print("\n  ❌ 问题：最终答案错了/空了，但你不知道是哪一步出了问题")
print("     是 step1 提取错了？step2 计算错了？还是 step3 格式化错了？")
print("     没有日志 = 黑盒")


# ══════════════════════════════════════════════════════════
# ② 加日志后：每步的输入/输出/耗时/状态一目了然
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 加日志后：每步可追踪")
print("=" * 60)

def run_with_logs(text, request_id):
    """带日志的运行"""
    logs = []
    t0 = time.time()

    # Step 1
    s1_start = time.time()
    r1 = step1_extract(text)
    s1_ms = (time.time() - s1_start) * 1000
    logs.append({"step": 1, "name": "extract", "input": text[:50],
                 "output": r1[:50], "ms": round(s1_ms), "status": "ok" if r1 else "empty"})
    print(f"  [{request_id}] Step1 extract: input={text[:30]!r} → output={r1!r} ({s1_ms:.0f}ms)")

    # Step 2
    s2_start = time.time()
    r2 = step2_calculate(r1)
    s2_ms = (time.time() - s2_start) * 1000
    status2 = "ok" if not r2.startswith("ERROR") else "error"
    logs.append({"step": 2, "name": "calculate", "input": r1[:50],
                 "output": r2[:50], "ms": round(s2_ms), "status": status2})
    print(f"  [{request_id}] Step2 calculate: input={r1!r} → output={r2!r} ({s2_ms:.0f}ms) {status2.upper()}")

    # Step 3
    s3_start = time.time()
    r3 = step3_format(r2)
    s3_ms = (time.time() - s3_start) * 1000
    logs.append({"step": 3, "name": "format", "input": r2[:50],
                 "output": r3[:50], "ms": round(s3_ms), "status": "ok" if r3 else "empty"})
    print(f"  [{request_id}] Step3 format: input={r2!r} → output={r3!r} ({s3_ms:.0f}ms)")

    total_ms = (time.time() - t0) * 1000
    print(f"\n  [{request_id}] 总耗时: {total_ms:.0f}ms, 最终答案: {r3[:60]}")

    # 保存结构化日志
    with open("/tmp/trace_log.jsonl", "a") as f:
        for log in logs:
            log["request_id"] = request_id
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

    return r3, logs


# 跑两个请求
print("\n--- 请求 1: 正常输入 ---")
r3, logs1 = run_with_logs("公司规定年假15天，病假30天。", "req-001")

time.sleep(GAP)
print("\n--- 请求 2: 会失败的输入 ---")
r3, logs2 = run_with_logs("年假和病假各有多少天？", "req-002")


# ══════════════════════════════════════════════════════════
# ③ 对比：有日志 vs 没日志
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 对比：有日志 vs 没日志")
print("=" * 60)

print("""
  没日志时：
    输入 "年假和病假各有多少天？" → 最终答案 "ERROR: invalid input" 被格式化成了什么？
    你不知道是 step1 提取错了，还是 step2 计算错了。

  有日志时：
    req-002 Step1 extract: input="年假和病假各有多少天？" → output="年假15天，病假30天"
    req-002 Step2 calculate: input="年假15天，病假30天" → output="ERROR: invalid input"  ← 一眼定位！
    req-002 Step3 format: input="ERROR: invalid input" → output="..."

  → **Step 2 是瓶颈**：输入不是纯数字，计算失败。
  → 修复：step2 应该先从 step1 的输出里提取数字，而不是直接 int()。
""")

print("结构化日志已写入 /tmp/trace_log.jsonl")
print("  → 每条记录包含: request_id, step, name, input, output, ms, status")
print("=" * 60)
