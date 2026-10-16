"""
W2 第 1 步：为什么需要 RAG —— 模型的知识边界

同一个问题，两种问法：
1. 裸问（模型不知道 → 会瞎编）
2. 把资料塞进 prompt（RAG 的核心思想 → 据实回答）
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))  # .env 在 week1

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)


def ask(messages, label):
    import time
    r = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="glm-4.7-flash",
                messages=messages,
                temperature=0.1,
                max_tokens=300,
                extra_body={"thinking": {"type": "disabled"}},
            )
            break
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  (限流，{2**attempt}s 后重试)")
            time.sleep(2 ** attempt)
    print("=" * 60)
    print(f"【{label}】")
    print("=" * 60)
    print(r.choices[0].message.content)
    print()


QUESTION = "我们公司规定，试用期员工转正需要提前多久提交申请？具体流程是什么？"

# ── 1. 裸问：模型不可能知道你公司的内部制度 ──
ask([{"role": "user", "content": QUESTION}], "① 裸问（不提供任何资料）")

# ── 2. 把"公司制度"原文塞进 prompt（RAG 的核心思想）──
policy = """【公司内部制度 · 节选】
试用期员工转正流程：
1. 员工需在试用期满前 15 个工作日，向直属上级提交《转正申请表》。
2. 直属上级在 5 个工作日内完成评估，并提交 HR。
3. HR 在 10 个工作日内完成审批，审批通过后次月 1 日正式生效。
注意：逾期未提交申请视为放弃本次转正资格。"""

ask([
    {"role": "system", "content": "你是公司制度问答助手，只能根据我提供的制度内容回答，不要编造。"},
    {"role": "user", "content": f"请阅读以下制度，然后回答问题。\n\n{policy}\n\n问题：{QUESTION}"},
], "② 把资料塞进 prompt（RAG 的核心思想）")
