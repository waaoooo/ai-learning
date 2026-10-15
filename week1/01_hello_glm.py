"""
Task 1（智谱版）：第一个真实 LLM API 调用
模型：glm-4.7-flash（免费）
"""
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "glm-4.7-flash"

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


def chat(messages, temperature=0.7, max_tokens=500):
    """基础调用，带重试"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}},  # 关掉推理内耗
            )
            return resp
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
            print(f"⚠️ 重试 {attempt+1}/2 ... ({e})")


def main():
    print("=" * 50)
    print(f"Task 1 · 真实调用 · 模型 {MODEL}")
    print("=" * 50)

    # 1. 基础调用
    print("\n--- 1. 基础调用 ---")
    resp = chat([{"role": "user", "content": "你好，用一句话介绍你自己"}])
    print("回复:", resp.choices[0].message.content)
    print("用量:", resp.usage)

    # 2. temperature 对比
    print("\n--- 2. temperature 对比（同一问题跑 3 次）---")
    for t in [0.1, 0.9]:
        resp = chat(
            [{"role": "user", "content": "用一句话描述春天"}],
            temperature=t,
        )
        print(f"temperature={t}: {resp.choices[0].message.content}")

    # 3. token 用量
    print("\n--- 3. token 用量 ---")
    resp = chat(
        [{"role": "user", "content": "用 200 字介绍人工智能的发展历史"}],
        max_tokens=600,
    )
    u = resp.usage
    print(f"prompt_tokens    : {u.prompt_tokens}")
    print(f"completion_tokens: {u.completion_tokens}")
    print(f"total_tokens     : {u.total_tokens}")
    print("成本: ¥0（flash 免费模型）")


if __name__ == "__main__":
    main()
