"""
W7 第 2 课：三类评测方法（续）

本课对比三种方法：
  ① 人工评测 —— 人读输出，打分/挑错
  ② 规则评测 —— 精确匹配/正则/关键词/格式校验
  ③ 模型评测 —— LLM-as-Judge 按标准打分

同一任务（年假问题），用三种方法评测同一个答案。
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

MODEL = "glm-4-flash"   # 非推理模型，最稳


def call(messages, temperature=0.0, system=None) -> str:
    last_err: Exception = RuntimeError("call failed")
    for attempt in range(4):
        try:
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            resp = client.chat.completions.create(
                model=MODEL, messages=msgs, temperature=temperature,
                max_tokens=200, extra_body={"thinking": {"type": "disabled"}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(3 * (2 ** attempt))
    raise last_err


# ══════════════════════════════════════════════════════════
# 准备：生成一个待评测的答案
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("准备：生成一个待评测的答案")
print("=" * 60)

Q = "年假15天，病假30天，请了3天年假+5天病假，还剩几天年假？"
print(f"\n  问题: {Q}\n")

# 模拟一个有问题的答案（格式正确，但计算错误）
TO_EVAL = "根据年假和病假的规定，请了3天年假和5天病假，所以剩余年假是15-3=12天。"
print(f"  待评测答案: {TO_EVAL}")

# 真正正确的答案
CORRECT = "12天"
print(f"  正确答案: {CORRECT}")


# ══════════════════════════════════════════════════════════
# ① 人工评测
# ═════════════════════════════════════剩余年假是12天，所以答案正确。但计算过程有误，病假不该减年假。答案正确，过程错误。评分：3/5（答案对但过程错）"
print("\n" + "=" * 60)
print("① 人工评测")
print("=" * 60)

RUBRIC = """
评分标准（1-5分）：
1分：完全错误，答非所问
2分：答案正确但过程严重错误
3分：答案正确但过程有误
4分：答案正确，过程基本正确
5分：答案正确，过程完美
"""

human_judgment = call(
    [{"role": "user", "content": f"任务：{Q}\n\n待评测答案：{TO_EVAL}\n\n{RUBRIC}\n\n请打分（1-5分）并写理由。"}],
    system="你是人工评委，严格按 rubric 评分。"
)

print(f"  人工评测结果:")
print(f"  评分: {human_judgment[:30]}...")
print("  → 优点：能判断「答案对但过程错」这种复杂情况")
print("  → 缺点：贵、慢、不同人标准不一")


# ══════════════════════════════════════════════════════════
# ② 规则评测
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 规则评测")
print("=" * 60)

# 精确匹配
exact_match = TO_EVAL == CORRECT
print(f"  精确匹配: {'✅ 通过' if exact_match else '❌ 失败'}")

# 关键词检查（答案包含"12天"）
keyword_check = "12天" in TO_EVAL
print(f"  关键词检查: {'✅ 通过' if keyword_check else '❌ 失败'}")

# 正则检查（答案包含数字+天）
pattern = r"\d+天"
regex_check = bool(re.search(pattern, TO_EVAL))
print(f"  正则检查(\\d+天): {'✅ 通过' if regex_check else '❌ 失败'}")

# 格式检查（答案以数字开头，结尾是"天"）
format_ok = re.match(r"^\d+天$", TO_EVAL.strip())
print(f"  格式检查(数字+天): {'✅ 通过' if format_ok else '❌ 失败'}")

# JSON 格式检查（如果要求输出 JSON）
try:
    json.loads(json.dumps({"answer": TO_EVAL}))
    json_ok = True
except json.JSONDecodeError:
    json_ok = False
print(f"  JSON格式检查: {'✅ 通过' if json_ok else '❌ 失败'}")

print("\n  → 规则评测结果:")
print("      精确匹配: ❌ 失败（但答案正确）")
print("      关键词检查: ✅ 通过（能抓住正确答案）")
print("      正则检查: ✅ 通过（格式合法）")
print("      JSON格式: ✅ 通过（能序列化）")
print("\n  → 优点：快、便宜、可复现")
print("  → 缺点：精确匹配会误判（答案正确但措辞不同）")


# ══════════════════════════════════════════════════════════
# ③ 模型评测（LLM-as-Judge）
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 模型评测（LLM-as-Judge）")
print("=" * 60)

JUDGE_PROMPT = f"""
你是一个评测员，任务是判断答案是否正确。
问题：{Q}
待评测答案：{TO_EVAL}
正确答案：{CORRECT}

评分标准：
- 1分：答案完全错误
- 2分：答案部分正确，但关键错误
- 3分：答案正确，但解释有误
- 4分：答案正确，解释基本正确
- 5分：答案完美，解释准确

请输出：
评分：X分
理由：...
"""

judge_result = call(
    [{"role": "user", "content": JUDGE_PROMPT}],
    system="你是严格的评测员，只输出评分和理由。"
)

print(f"  模型评测结果:")
print(f"  {judge_result}")

print("\n  → 优点：能理解语义，比规则更灵活")
print("  → 缺点：可能有偏见，裁判本身也会错")


# ══════════════════════════════════════════════════════════
# 三层混合
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("三层混合策略")
print("=" * 60)

print("""
第一层：规则评测（快/便宜/可复现）
  ✅ 格式合法（正则通过）
  ✅ 关键词正确（包含"12天"）
  ✅ JSON能解析
  → 自动通过，无需人工

第二层：模型评测（能理解语义）
  ⚠️ LLM-as-Judge 给出3分（答案正确但解释有误）
  → 标记"需人工复核"

第三层：人工抽检（最准，但最贵）
  抽检 10% 的模型打分结果
  → 重点复核"答案正确但过程错"的情况

三层混合：每次改代码都跑规则 + 模型评测，定期人工抽检。
""")
print("=" * 60)
