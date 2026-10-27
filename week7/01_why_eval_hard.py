"""
W7 第 1 课：为什么评测这么难

核心问题：LLM 输出不确定 —— 同一输入，不同输出。
传统测试"固定输入→固定输出→断言相等"在这里失效。

本课用真实调用演示：
  ① 同一问题问 3 次（temperature=0.7）→ 答案不同
  ② 传统"精确匹配"测试如何失败
  ③ temperature=0 能降低方差，但降不到零
"""
import os
import time
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)   # ★ 加超时，避免无限等待

N = 3          # 每组调用次数（免费额度限流，别调大）
GAP = 4.0      # 每次调用间隔（秒）
MODEL = "glm-4-flash"   # 非推理模型，最稳


def call(question, temperature=0.0, system=None) -> str:
    last_err: Exception = RuntimeError("call failed")
    for attempt in range(4):
        try:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": question}]
            resp = client.chat.completions.create(
                model=MODEL, messages=msgs, temperature=temperature,
                max_tokens=120, extra_body={"thinking": {"type": "disabled"}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(3 * (2 ** attempt))     # 限流时退避更久
    raise last_err


def batch(question, temperature, n=N):
    """连问 n 次，每次间隔"""
    out = []
    for i in range(n):
        out.append(call(question, temperature))
        print(f"      …第 {i+1}/{n} 次完成", flush=True)
        time.sleep(GAP)
    return out


# ══════════════════════════════════════════════════════════
# ① 同一问题问 N 次（temperature=0.7）
# ══════════════════════════════════════════════════════════
print("=" * 60)
print(f"① 同一问题问 {N} 次（temperature=0.7）")
print("=" * 60)

Q1 = "用一句话介绍什么是 RAG？"
print(f"\n  问题: {Q1}\n")

answers_hi = batch(Q1, temperature=0.7)
for i, a in enumerate(answers_hi, 1):
    print(f"  第{i}次: {a}")

unique_hi = len(set(answers_hi))
print(f"\n  → {N} 次调用，得到 {unique_hi} 种不同答案")


# ══════════════════════════════════════════════════════════
# ② 传统"精确匹配"测试如何失败
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 传统测试方法在这里失效")
print("=" * 60)

EXPECTED = "RAG 是检索增强生成。"

print(f"\n  假设标准答案: 「{EXPECTED}」")
print(f"  传统断言: answer == EXPECTED\n")
passed = sum(1 for a in answers_hi if a == EXPECTED)
for i, a in enumerate(answers_hi, 1):
    print(f"    第{i}次: {'✅ 通过' if a == EXPECTED else '❌ 失败'}")
print(f"\n  → 通过率: {passed}/{N}")
print("  ❌ 但这是**假失败** —— 那些答案其实都对，只是措辞不同")


# ══════════════════════════════════════════════════════════
# ③ temperature 的影响
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ temperature 能降低方差，但降不到零")
print("=" * 60)

Q2 = "公司年假有多少天？"
print(f"\n  问题: {Q2}\n")

for temp in (0.0, 0.7):
    ans = batch(Q2, temperature=temp)
    counter = Counter(ans)
    print(f"  temperature={temp}:")
    for a, c in counter.most_common():
        print(f"    「{a[:40]}」 × {c}")
    print(f"    → {len(counter)} 种不同答案\n")


# ══════════════════════════════════════════════════════════
# ④ 结论
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("结论：LLM 评测为什么难")
print("=" * 60)
print("""
  ① 输出不确定 —— 同一问题多次调用，措辞就不同
  ② 没有唯一正确答案 —— "RAG 是什么"有无数种正确表述
  ③ 传统"精确匹配"会大量误判 —— 把对的答案判成错的
  ④ temperature=0 只能降低方差，不能消除

  → 所以需要新的评测方法：
     不是"答案是否等于标准答案"，
     而是"答案是否**满足要求**"。
""")
print("=" * 60)
