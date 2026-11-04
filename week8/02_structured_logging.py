"""
W8 第 2 课：结构化日志与 request_id

演示：
  ① 文本日志 vs JSON 日志（对比可读性和可查询性）
  ② request_id 从头到尾传递（链路追踪的基石）
  ③ 用 logging 模块 + JSON formatter 实现结构化日志
"""
import os
import json
import time
import uuid
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 3.0


# ══════════════════════════════════════════════════════════
# ① 文本日志 vs JSON 日志
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 文本日志 vs JSON 日志")
print("=" * 60)

print("""
  文本日志（人类友好，机器难用）：
    2026-10-28 10:30:01 INFO  req-001 Step1 extract input=公司规定年假15天... output=15\\n30 (2266ms)
    2026-10-28 10:30:03 ERROR req-001 Step2 calculate input=15\\n30 output=ERROR: invalid input (0ms)

  问题：
    - 字段位置不固定 → 解析困难
    - input/output 里有空格/换行 → 正则匹配容易出错
    - 无法高效查询（"找出所有 ERROR"需要 grep + 正则）

  JSON 日志（机器友好，人类也能读）：
    {"ts":"2026-10-28T10:30:01","req_id":"req-001","step":1,"name":"extract",
     "input":"公司规定年假15天...","output":"15\\n30","ms":2266,"status":"ok"}
    {"ts":"2026-10-28T10:30:03","req_id":"req-001","step":2,"name":"calculate",
     "input":"15\\n30","output":"ERROR: invalid input","ms":0,"status":"error"}

  优势：
    - 每条都是独立 JSON → 用 jq 一行命令就能查
    - 固定字段 → 程序自动解析，不会因格式变化而崩溃
    - 可直接导入 ELK/Grafana/Loki 做可视化
""")

# ══════════════════════════════════════════════════════════
# ② request_id 传递：链路追踪的基石
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("② request_id 传递：链路追踪的基石")
print("=" * 60)


class TraceLogger:
    """结构化日志器：每条日志自动带上 request_id"""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.logs = []

    def log(self, step: int, name: str, input_text: str, output: str,
            ms: float, status: str = "ok", **extra):
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "req_id": self.request_id,
            "step": step,
            "name": name,
            "input": input_text[:100],      # 截断避免日志过大
            "output": output[:100],
            "ms": round(ms),
            "status": status,
        }
        entry.update(extra)   # 可扩展字段（token_count, model, etc.）
        self.logs.append(entry)
        # 用 json.dumps 输出，ensure_ascii=False 保留中文
        print(f"  {json.dumps(entry, ensure_ascii=False)}")

    def dump(self, filepath: str):
        with open(filepath, "a") as f:
            for entry in self.logs:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def call_llm(prompt: str) -> str:
    last_err: Exception = RuntimeError("call failed")
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=150,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err


def run_chain(text: str, tracer: TraceLogger):
    """带追踪的 3 步链路"""
    # Step 1: 提取
    t0 = time.time()
    r1 = call_llm(f"从以下文本提取年假天数：{text}\n只输出数字。")
    tracer.log(1, "extract", text, r1, (time.time() - t0) * 1000,
               status="ok" if r1 else "empty")

    # Step 2: 计算
    t0 = time.time()
    try:
        days = int(r1.strip().split("\n")[0])
        r2 = str(days - 3)
        status2 = "ok"
    except ValueError:
        r2 = "ERROR: invalid input"
        status2 = "error"
    tracer.log(2, "calculate", r1, r2, (time.time() - t0) * 1000, status=status2)

    # Step 3: 格式化
    t0 = time.time()
    r3 = call_llm(f"把以下结果格式化成友好回复：{r2}\n输出一句话。")
    tracer.log(3, "format", r2, r3, (time.time() - t0) * 1000,
               status="ok" if r3 else "empty", model=MODEL)

    return r3


# ══════════════════════════════════════════════════════════
# ③ 实战：两个请求，request_id 互不混淆
# ══════════════════════════════════════════════════════════
print("\n--- 请求 A ---")
tracer_a = TraceLogger("req-A")
result_a = run_chain("公司规定年假15天，病假30天。", tracer_a)
tracer_a.dump("/tmp/trace_structured.jsonl")

time.sleep(GAP)
print("\n--- 请求 B ---")
tracer_b = TraceLogger("req-B")
result_b = run_chain("年假和病假各有多少天？", tracer_b)
tracer_b.dump("/tmp/trace_structured.jsonl")

# ══════════════════════════════════════════════════════════
# ④ 查询：用 jq 一行命令就能查
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 查询结构化日志（jq 示例）")
print("=" * 60)
print("""
  # 找出所有 ERROR 状态的日志
  cat /tmp/trace_structured.jsonl | jq 'select(.status=="error")'

  # 找出 req-A 的 Step 2
  cat /tmp/trace_structured.jsonl | jq 'select(.req_id=="req-A" and .step==2)'

  # 找出耗时超过 2 秒的步骤
  cat /tmp/trace_structured.jsonl | jq 'select(.ms > 2000)'

  # 统计每个步骤的平均耗时
  cat /tmp/trace_structured.jsonl | jq -s 'group_by(.name) | map({name: .[0].name, avg_ms: (map(.ms)|add/length)})'

  → 文本日志做不到这些，JSON 日志一行命令搞定。
""")

print("=" * 60)
print("关键设计要点：")
print("  1. request_id 从入口生成，传给所有下游步骤")
print("  2. 每条日志固定字段：ts, req_id, step, name, input, output, ms, status")
print("  3. input/output 截断（避免日志过大）")
print("  4. JSON 格式 → 可被 jq/ELK/Grafana 直接消费")
print("  5. 可扩展字段（model, token_count, temperature）")
print("=" * 60)
