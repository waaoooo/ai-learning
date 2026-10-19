"""
W4 第 3 课：自我反思（Self-Reflection）

生成答案后，再调一次模型让它当"批评者"检查自己：
  有没有遗漏？有没有错误？需要修正吗？
"""
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

def call(messages, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    for attempt in range(4):
        try:
            return client.chat.completions.create(
                model="glm-4.7-flash",
                messages=msgs,
                temperature=0,
                max_tokens=600,
                extra_body={"thinking": {"type": "disabled"}},
            ).choices[0].message.content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


# ══════════════════════════════════════════════════════════
# ① 生成（第一次回答）
# ══════════════════════════════════════════════════════════
TASK = "北京今天天气怎么样？适合穿什么衣服出门？"

print("=" * 60)
print("① 生成阶段")
print("=" * 60)
print(f"\n任务: {TASK}\n")

WEATHER = "北京天气：晴，22℃，东风3级"
draft = call(
    [{"role": "user", "content": f"天气数据：{WEATHER}\n\n请回答用户问题：{TASK}"}],
    system="你是助手，根据天气数据回答用户问题。",
)
print(f"初稿:\n{draft}\n")


# ══════════════════════════════════════════════════════════
# ② 反思（让模型当批评者，检查自己的输出）
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("② 反思阶段（模型检查自己）")
print("=" * 60)

REFLECT_SYSTEM = """你是一个严格的质检员。检查下面的回答是否完整、正确地解决了用户任务。
重点检查：
1. 任务的每个部分都回答了吗？（有没有漏掉问题）
2. 事实和计算有没有错误？
3. 有没有编造数据？

如果完全正确，只输出：PASS
如果有问题，输出 JSON：{"ok": false, "problems": ["问题1", "问题2"], "fix": "应该怎么改"}"""

critique = call(
    [{"role": "user", "content": f"任务：{TASK}\n\n已知信息：{WEATHER}\n\n待检查的回答：\n{draft}"}],
    system=REFLECT_SYSTEM,
)
print(f"\n反思结果:\n{critique}\n")


# ══════════════════════════════════════════════════════════
# ③ 修正（有问题才改）
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("③ 修正阶段")
print("=" * 60)

if critique.strip().upper().startswith("PASS"):
    print("\n✅ 检查通过，无需修正")
    final = draft
else:
    try:
        info = json.loads(critique)
    except json.JSONDecodeError:
        info = {"fix": critique}
    print(f"\n⚠️ 发现问题: {info.get('problems', [])}")
    print(f"   修正方向: {info.get('fix', '')}\n")
    final = call(
        [{"role": "user", "content": f"任务：{TASK}\n\n原回答：{draft}\n\n指出的问题：{info}"}],
        system="你是助手，根据指出的问题修正回答，确保完整正确。",
    )
    print(f"修正后:\n{final}\n")

print("=" * 60)
print(f"最终输出:\n{final}")
print("=" * 60)
print("关键：生成 → 反思（质检）→ 修正。让模型自己当自己的批评者。")
