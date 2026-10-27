"""
W6 第 1 课：为什么需要多 Agent

单 Agent 的三个天花板：
  ① 上下文天花板 —— 一个窗口装不下所有资料（数学演示）
  ② 专业天花板   —— 一个 prompt 身兼数职，每个角色都被稀释
  ③ 可靠性天花板 —— 没有交叉校验，错误直接传导（真实 LLM 演示）
"""
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/")

# 中文大致 1 字 ≈ 1 token（保守按 1.5 估）
CHARS_PER_TOKEN = 1.5


def call(messages, system=None):
    for attempt in range(4):
        try:
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            return client.chat.completions.create(
                model="glm-4.7-flash", messages=msgs, temperature=0, max_tokens=400,
                extra_body={"thinking": {"type": "disabled"}},
            ).choices[0].message.content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


# ══════════════════════════════════════════════════════════
# ① 上下文天花板：单 Agent 装不下所有资料
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 上下文天花板：一个 Agent 装不下所有资料")
print("=" * 60)

# 模拟一个真实项目要喂给 Agent 的资料
docs = {
    "需求文档": "用户希望做一个企业内部的智能问答系统，支持上传 PDF/Word，"
              "能按部门隔离权限，支持多轮追问，要求回答带引用来源。" * 50,
    "架构文档": "系统分为接入层、编排层、资源层。接入层负责鉴权和限流；"
              "编排层负责任务规划和工具调度；资源层负责知识库和工具调用。" * 50,
    "数据库设计": "用户表、文档表、切片表、会话表、消息表。切片表需要记录原文、"
                "向量、所属文档、权限范围。" * 50,
    "API 设计": "POST /ask 接收问题返回答案；POST /upload 上传文档；"
              "GET /history 返回会话历史；DELETE /doc 删除文档。" * 50,
    "测试用例": "需要覆盖：空文档、超大文档、权限越界、并发上传、检索无结果等场景。" * 50,
    "部署方案": "使用 Docker Compose 编排，前端 Nginx，后端 FastAPI，"
              "向量库独立部署，配置中心管理密钥。" * 50,
}

total_chars = sum(len(v) for v in docs.values())
total_tokens = int(total_chars / CHARS_PER_TOKEN)
WINDOW = 8000

print(f"\n  资料份数: {len(docs)} 份")
print(f"  总字符数: {total_chars}")
print(f"  估算 token: {total_tokens}")
print(f"  假设窗口: {WINDOW} token")
print(f"  → 装得下吗: {'❌ 装不下' if total_tokens > WINDOW else '✅ 装得下'}")

if total_tokens > WINDOW:
    print(f"  → 溢出: {total_tokens - WINDOW} token（{total_tokens/WINDOW:.1f} 倍）")
    print("  ❌ 单 Agent 无法一次性处理全部资料 → 必须分工/分阶段")

print("\n  💡 多 Agent 解法：每个 Agent 只拿它需要的那部分资料")
for name, v in docs.items():
    t = int(len(v) / CHARS_PER_TOKEN)
    print(f"      {name}: {t} token" + ("  ← 仍超窗口" if t > WINDOW else ""))


# ══════════════════════════════════════════════════════════
# ② 专业天花板：一个 prompt 身兼数职
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 专业天花板：一个 Agent 身兼数职 vs 分工")
print("=" * 60)

TASK = "帮我评估这个需求：'给问答系统加一个语音输入功能'，从产品和工程两个角度分析。"

print(f"\n  任务: {TASK}\n")

# A：一个 Agent 身兼数职
generic = call([{"role": "user", "content": TASK}],
               system="你是一个全能助手，什么都懂一点。")
print("  【A】单 Agent（全能但都不精）:")
print("  " + generic.replace("\n", "\n  ")[:600])

# B：两个专业 Agent 分工
print("\n  【B】多 Agent（产品 + 工程分工）:")
product = call([{"role": "user", "content": TASK}],
               system="你是一位资深产品经理，只从产品角度分析：用户价值、场景、优先级、风险。不要谈技术实现。")
engineer = call([{"role": "user", "content": TASK}],
                system="你是一位资深工程师，只从工程角度分析：技术方案、工作量、风险、依赖。不要谈产品价值。")
print("  ── 产品 Agent ──")
print("  " + product.replace("\n", "\n  ")[:400])
print("\n  ── 工程 Agent ──")
print("  " + engineer.replace("\n", "\n  ")[:400])

print("\n  💡 分工后：每个角色都更专注、更深入，不会互相稀释。")


# ══════════════════════════════════════════════════════════
# ③ 可靠性天花板：没有交叉校验
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 可靠性天花板：单 Agent 的错误无人发现")
print("=" * 60)

TRAP = ("公司年假15天，病假每年30天。小王请了3天年假和5天病假，"
        "请问他还剩多少天年假？")

print(f"\n  题目: {TRAP}\n")
single = call([{"role": "user", "content": TRAP}],
              system="你是一个助手，直接给出答案。")
print("  【A】单 Agent 直接回答:")
print("  " + single.replace("\n", "\n  ")[:300])

print("\n  【B】增加一个'校验 Agent'复核:")
verdict = call([{"role": "user", "content": f"请检查下面的解答是否正确，指出任何错误。\n\n题目：{TRAP}\n\n解答：{single}"}],
               system="你是一位严格的数学和逻辑校验员。只检查事实和计算，发现错误就明确指出并给出正确答案。")
print("  " + verdict.replace("\n", "\n  ")[:500])

print("\n" + "=" * 60)
print("三个天花板 → 三个多 Agent 的必要性：")
print("  ① 上下文装不下 → 分工处理不同资料")
print("  ② 专业被稀释   → 分工扮演不同角色")
print("  ③ 错误无校验   → 分工做交叉复核")
print("=" * 60)
