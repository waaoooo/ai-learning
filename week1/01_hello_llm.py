"""
Task 1: 环境搭建 + 第一个 API 调用
目标：成功调用 DeepSeek API，理解请求/响应结构
"""
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def chat(
    messages: list[dict],
    model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: int = 500
) -> str:
    """基础调用函数，带重试机制"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # 指数退避
            print(f"⚠️ 调用失败，{wait_time}s 后重试... ({e})")
            time.sleep(wait_time)

def main():
    print("=" * 50)
    print("Task 1: 第一个 LLM API 调用")
    print("=" * 50)
    
    # 基础调用
    print("\n--- 基础调用 ---")
    response = chat([
        {"role": "user", "content": "你好，请用一句话介绍你自己"}
    ])
    print(f"回复: {response}")
    
    # 对比 temperature
    print("\n--- Temperature 对比 ---")
    prompt = "用一句话描述春天"
    
    for temp in [0.1, 0.5, 0.9]:
        response = chat(
            [{"role": "user", "content": prompt}],
            temperature=temp
        )
        print(f"temperature={temp}: {response}")
    
    # Token 用量统计
    print("\n--- Token 用量 ---")
    response_full = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "写一篇 500 字的文章，介绍人工智能的发展历史"}],
        max_tokens=600
    )
    usage = response_full.usage
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    print(f"预估成本: ¥{usage.prompt_tokens * 0.000001 + usage.completion_tokens * 0.000002:.4f}")

if __name__ == "__main__":
    main()
