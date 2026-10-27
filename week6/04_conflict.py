"""
W6 第 4 课：冲突与汇总

场景：产品 Agent 和工程 Agent 结论矛盾
  ① 权威裁决 —— 按决策领域选对应专家
  ② 第三方仲裁 —— 引入中立评估 Agent
  ③ 升级人工   —— 无法调和时报给决策者
"""
import os
import json
import time
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
                model="glm-4.7-flash", messages=msgs, temperature=0, max_tokens=350,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err


TASK = "给问答系统加语音输入功能"

print("=" * 60)
print("场景：两个 Agent 结论矛盾")
print("=" * 60)

# 两个 Agent 各自给出结论
product_out = call(
    [{"role": "user", "content": f"任务：{TASK}\n\n只做产品判断：这个功能该不该做？给明确结论。"}],
    system="你是资深产品经理。")
engineer_out = call(
    [{"role": "user", "content": f"任务：{TASK}\n\n假设：团队只有 2 人，本季度剩余工期 1 个月，ASR 云服务费每年 10 万。"
                                  f"只做工程判断：这个季度能不能上线？给明确结论。"}],
    system="你是资深工程师。")

print(f"\n  产品 Agent: {product_out.strip()[:120]}")
print(f"\n  工程 Agent: {engineer_out.strip()[:120]}")

conflict = {"product": product_out, "engineer": engineer_out}


# ══════════════════════════════════════════════════════════
# ① 权威裁决：按"决策领域"选专家
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("① 权威裁决：先问「这是谁的决策领域」")
print("=" * 60)

DOMAIN_RULES = {
    "要不要做（价值/优先级）": "product_agent",     # 产品决策 → 听产品
    "能不能做（技术/工期）": "engineer_agent",      # 技术决策 → 听工程
    "会不会出事（风险/合规）": "tester_agent",      # 风险决策 → 听测试
}

decision = call(
    [{"role": "user", "content": f"任务：{TASK}\n\n两个 Agent 的分歧：\n产品：{product_out}\n工程：{engineer_out}\n\n"
                                  f"现在要决定的是：「本季度是否排期做这个功能」。请判断这属于哪个决策领域"
                                  f"（要不要做 / 能不能做 / 会不会出事），并给出裁决：听谁的，为什么。"}],
    system="你是项目调度者，负责裁决分歧。",
)
print("\n  裁决过程:")
print("  " + decision.replace("\n", "\n  ")[:400])


# ══════════════════════════════════════════════════════════
# ② 第三方仲裁：中立 Agent 综合评估
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 第三方仲裁：引入中立评估 Agent")
print("=" * 60)

arbitration = call(
    [{"role": "user", "content": f"任务：{TASK}\n\n产品 Agent 说：{product_out}\n\n工程 Agent 说：{engineer_out}\n\n"
                                  f"请你作为中立的第三方，综合双方观点，给出：1) 双方合理之处 2) 折中方案 3) 最终建议。"}],
    system="你是一位中立的资深项目顾问，不偏向任何一方，只讲事实和权衡。",
)
print("\n  仲裁结果:")
print("  " + arbitration.replace("\n", "\n  ")[:500])


# ══════════════════════════════════════════════════════════
# ③ 升级人工：无法调和时报给决策者
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 升级人工：Agent 可以分析，但不替人拍板")
print("=" * 60)

escalation = call(
    [{"role": "user", "content": f"任务：{TASK}\n\n产品：{product_out}\n\n工程：{engineer_out}\n\n"
                                  f"假设仲裁也无法达成一致。请写一份给决策者（老板）的升级报告："
                                  f"1) 分歧焦点 2) 各方案的成本与收益 3) 需要老板拍板的问题。"}],
    system="你是项目调度者，负责把无法调和的分歧整理成决策报告，不替老板做决定。",
)
print("\n  升级报告:")
print("  " + escalation.replace("\n", "\n  ")[:500])


print("\n" + "=" * 60)
print("三种策略的适用边界：")
print("  ① 权威裁决 —— 问题边界清晰，知道该听谁的")
print("  ② 第三方仲裁 —— 双方都有理，需要权衡折中")
print("  ③ 升级人工   —— 涉及战略/成本取舍，Agent 不该替人拍板")
print("=" * 60)
