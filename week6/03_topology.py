"""
W6 第 3 课：两种拓扑实测

拓扑A 流水线（Pipeline）：A → B → C 顺序接力，前者产出是后者输入
拓扑B 中心调度（Hub）    ：调度者同时派发给所有 Agent，最后汇总
"""
import os
import json
import time
import uuid
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/")


def call(messages, system=None) -> str:
    last_err: Exception = RuntimeError("call failed")
    for attempt in range(4):
        try:
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            resp = client.chat.completions.create(
                model="glm-4.7-flash", messages=msgs, temperature=0, max_tokens=300,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err


TASK = "评估需求：给问答系统加一个语音输入功能。"

ROLES = {
    "product":  "你是资深产品经理，只从产品角度分析：用户价值、场景、优先级、风险。",
    "engineer": "你是资深工程师，只从工程角度分析：技术方案、工作量、风险、依赖。",
    "tester":   "你是资深测试工程师，只从测试角度分析：测试点、风险场景、验收标准。",
}


def agent_run(role_key, task, input_text=""):
    """单个 Agent 执行一次"""
    prompt = f"【角色】{ROLES[role_key]}\n\n【任务】{task}"
    if input_text:
        prompt += f"\n\n【上游产出】{input_text}"
    return call([{"role": "user", "content": prompt}], system=ROLES[role_key])


# ══════════════════════════════════════════════════════════
# 拓扑A：流水线（顺序接力）
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("拓扑A：流水线（Pipeline）")
print("=" * 60)
print(f"\n  任务: {TASK}\n")

t0 = time.time()
chain = []
input_text = ""
for role in ("product", "engineer", "tester"):
    print(f"  ── {role} ──")
    out = agent_run(role, TASK, input_text)
    print(f"  {out[:100].replace(chr(10),' ')}...\n")
    chain.append({"role": role, "output": out[:60]})
    input_text = out          # ★ 前者产出 → 后者输入
elapsed_a = time.time() - t0

print(f"  ⏱ 耗时: {elapsed_a:.1f}s  步骤数: {len(chain)}（串行）")


# ══════════════════════════════════════════════════════════
# 拓扑B：中心调度（Hub）
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("拓扑B：中心调度（Hub）")
print("=" * 60)

t0 = time.time()
# 调度者同时派发给三个 Agent（这里顺序执行，但逻辑上是并行的）
results = {}
for role in ("product", "engineer", "tester"):
    print(f"  ── 派发给 {role} ──")
    results[role] = agent_run(role, TASK)
    print(f"  {results[role][:100].replace(chr(10),' ')}...\n")

# 调度者汇总
summary = call(
    [{"role": "user", "content": f"任务：{TASK}\n\n以下是三个角色的独立分析：\n\n{json.dumps(results, ensure_ascii=False)}"
                                 f"\n\n请汇总成一份给决策者的简报，并明确指出有分歧的地方。"}],
    system="你是项目总调度，负责汇总各方意见并指出分歧。",
)
elapsed_b = time.time() - t0

print("  ── 调度者汇总 ──")
print("  " + summary.replace("\n", "\n  ")[:400])
print(f"\n  ⏱ 耗时: {elapsed_b:.1f}s  步骤数: {len(results)} + 1（汇总）")


# ══════════════════════════════════════════════════════════
# 对比
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("对比")
print("=" * 60)
print(f"""
  ┌──────────────┬─────────────────────┬─────────────────────┐
  │              │ A 流水线            │ B 中心调度          │
  ├──────────────┼─────────────────────┼─────────────────────┤
  │ 数据流       │ 顺序接力，有依赖     │ 各自独立，无依赖     │
  │ 可否并行     │ ❌ 不能（有依赖）    │ ✅ 能（互相独立）    │
  │ 上游影响下游 │ ✅ 会（后者能看到前者）│ ❌ 不会（互相隔离）  │
  │ 故障传播     │ ✅ 会（一步错步步错）│ ❌ 不会（各自隔离）  │
  │ 最终汇总     │ 天然线性，无需汇总   │ 需要调度者汇总       │
  │ 适合         │ 有先后依赖的流程     │ 多角度独立分析       │
  └──────────────┴─────────────────────┴─────────────────────┘

  实测耗时: 流水线 {elapsed_a:.1f}s  vs  中心调度 {elapsed_b:.1f}s
""")
print("=" * 60)
print("关键：有依赖用流水线，要独立视角用中心调度。")
print("=" * 60)
