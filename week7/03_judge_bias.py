"""
W7 第 3 课：LLM-as-Judge 的偏差与校准

本课用真实调用实测裁判的三类偏差：
  ① 位置偏差 —— 同一对答案，交换顺序重测，看裁判是否"跟着位置走"
  ② 长度偏差 —— 短而正确 vs 长而错误，看裁判是否偏爱长的
  ③ 自我偏好 —— 用不同模型当裁判，看结论是否一致

校准手段（本课同时验证）：
  - 交换顺序重测（bidirectional）
  - 多数投票
  - 明确 rubric + 要求"先给理由再给结论"
"""
import os
import re
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)

MODEL = "glm-4-flash"
JUDGE_MODEL = "glm-4.5-flash"   # 换一个模型当裁判，降低自我偏好

GAP = 4.0


def call(messages, temperature=0.0, system=None, model=MODEL) -> str:
    last_err: Exception = RuntimeError("call failed")
    for attempt in range(4):
        try:
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            resp = client.chat.completions.create(
                model=model, messages=msgs, temperature=temperature,
                max_tokens=200, extra_body={"thinking": {"type": "disabled"}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(3 * (2 ** attempt))
    raise last_err


def ask_judge(question, cand_a, cand_b, judge_model=JUDGE_MODEL):
    """让裁判在两个候选中选一个，返回 ('A'/'B'/'?', 原始输出)"""
    prompt = (
        "问题：" + question + "\n\n"
        "候选答案 A：\n" + cand_a + "\n\n"
        "候选答案 B：\n" + cand_b + "\n\n"
        "判断标准：只比较「答案是否正确、解释是否准确」，不考虑长度或措辞风格。\n\n"
        "请严格按以下 JSON 格式输出，不要输出任何其他内容：\n"
        '{"winner": "A 或 B", "reason": "一句话理由"}'
    )
    r = call([{"role": "user", "content": prompt}],
             system="你是严格的评测裁判，只输出 JSON。", model=judge_model)
    time.sleep(GAP)

    # ① 首选：解析 JSON 里的 winner 字段
    m = re.search(r'"winner"\s*:\s*"([ABab])"', r)
    if m:
        return m.group(1).upper(), r

    # ② 回退：从最后一行往前找独立出现的 A 或 B
    lines = [l.strip() for l in r.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        up = line.upper()
        a = re.search(r"(?<![A-Z])A(?![A-Z])", up)
        b = re.search(r"(?<![A-Z])B(?![A-Z])", up)
        if a and not b:
            return "A", r
        if b and not a:
            return "B", r
    return "?", r


# ══════════════════════════════════════════════════════════
# 测试数据：A 正确、B 错误
# ══════════════════════════════════════════════════════════
Q = "年假15天，病假30天，请了3天年假+5天病假，还剩几天年假？"

A_CORRECT = "12天。年假15天，请了3天年假，15-3=12天。病假不影响年假。"
B_WRONG = "7天。因为总共请了3+5=8天假，15-8=7天。"

print("=" * 60)
print("测试数据")
print("=" * 60)
print(f"\n  问题: {Q}")
print(f"  A（正确）: {A_CORRECT}")
print(f"  B（错误）: {B_WRONG}")


# ══════════════════════════════════════════════════════════
# ① 位置偏差：交换顺序重测
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("① 位置偏差：交换顺序重测")
print("=" * 60)

r1, raw1 = ask_judge(Q, A_CORRECT, B_WRONG)   # 顺序 [A, B]
print(f"\n  顺序 [A, B] → 裁判选: {r1}")
print(f"    理由: {raw1.split(chr(10))[0][:80]}")

r2, raw2 = ask_judge(Q, B_WRONG, A_CORRECT)   # 顺序 [B, A]
print(f"\n  顺序 [B, A] → 裁判选: {r2}")
print(f"    理由: {raw2.split(chr(10))[0][:80]}")

print("\n  判定:")
if r1 == "A" and r2 == "B":
    print("  ✅ 裁判两次都选 A（正确答案）→ **无位置偏差，可靠**")
elif r1 == "B" and r2 == "A":
    print("  ❌ 裁判两次都选 B（错误答案）→ **裁判判断力有问题**")
elif r1 == r2:
    print(f"  ⚠️ 两次都选 {r1} → **位置偏差**（裁判跟着位置走，不是跟着内容走）")
else:
    print(f"  ⚠️ 结论不一致（{r1} vs {r2}）→ 裁判不稳定，需多数投票")


# ══════════════════════════════════════════════════════════
# ② 长度偏差：短而正确 vs 长而错误
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 长度偏差：短而正确 vs 长而错误")
print("=" * 60)

SHORT_CORRECT = "12天。"
LONG_WRONG = ("我们来仔细分析一下这个问题。首先，我们需要理解题目给出的条件：年假有15天，"
              "病假有30天。然后，题目说请了3天年假和5天病假。"
              "那么，一个常见的思路是把所有假期加起来计算，也就是 3+5=8 天。"
              "接下来，用年假总数减去已请的假：15-8=7天。"
              "所以，剩余的年假是7天。这个计算过程非常清晰，逻辑也很完整，"
              "因此我们有理由相信这个答案是正确的。")

print(f"\n  短答案（正确）: {SHORT_CORRECT}")
print(f"  长答案（错误）: {LONG_WRONG[:60]}…")

r3, raw3 = ask_judge(Q, SHORT_CORRECT, LONG_WRONG)
print(f"\n  顺序 [短正确, 长错误] → 裁判选: {r3}")
print(f"    理由: {raw3.split(chr(10))[0][:80]}")

if r3 == "A":
    print("  ✅ 裁判选了短的正确答案 → **无长度偏差**")
else:
    print("  ❌ 裁判选了长而错误的答案 → **长度偏差**（被篇幅迷惑）")


# ══════════════════════════════════════════════════════════
# ③ 校准：多数投票 + 多裁判
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 校准手段：交换顺序 + 多数投票")
print("=" * 60)

judges = [("glm-4.5-flash", "J1"), ("glm-4.7-flash", "J2"), ("glm-4-flash", "J3")]
votes = []
for jm, jn in judges:
    v, _ = ask_judge(Q, A_CORRECT, B_WRONG, judge_model=jm)
    votes.append(v)
    print(f"  {jn} ({jm}): 选 {v}")

from collections import Counter
c = Counter(votes)
winner, n = c.most_common(1)[0]
print(f"\n  多数投票结果: {winner}（{n}/{len(votes)} 票）")
print(f"  → {'✅ 多数票选对（A）' if winner == 'A' else '❌ 多数票选错'}")
print("\n  → 多数投票能抵消单个裁判的偏差，但**不能抵消系统性偏差**")
print("    如果三个裁判都偏爱长答案，多数投票只会放大这个偏差。")
print("    所以：多数投票解决「随机误差」，**rubric 解决「系统性偏差」**。")


print("\n" + "=" * 60)
print("结论：校准 LLM-as-Judge 的手段")
print("=" * 60)
print("""
  ① 交换顺序重测（bidirectional）—— 揪出位置偏差
  ② 明确 rubric + 先理由后结论 —— 减少随意性
  ③ 多数投票 —— 抵消单个裁判的随机误差
  ④ 换模型当裁判 —— 降低自我偏好
  ⑤ 人工抽检 10% —— 校准裁判本身

  核心认知：多数投票解决随机误差，rubric 解决系统性偏差。
""")
print("=" * 60)
