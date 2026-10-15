"""
Task 1 (Mock 版本): 无 API Key 也能跑通
使用模拟响应，帮助理解代码结构
"""
import os
import time
import json
from unittest.mock import MagicMock, patch

# ============ Mock 响应 ============
MOCK_RESPONSES = {
    "你好，请用一句话介绍你自己": "我是 DeepSeek，一个由深度求索公司开发的人工智能助手。",
    "用一句话描述春天": {
        0.1: "春天是万物复苏的季节，气温逐渐回暖。",
        0.5: "春天来了，大地披上了绿装，花儿绽放，鸟儿歌唱。",
        0.9: "春天就像一位神奇的画家，轻轻一挥笔，整个世界就变成了五彩斑斓的画卷！"
    },
    "写一篇 500 字的文章，介绍人工智能的发展历史": 
        "人工智能的发展可以追溯到 1956 年的达特茅斯会议。经过几十年的起伏，"
        "特别是近年来深度学习的突破，AI 已经渗透到我们生活的方方面面。"
        "从语音识别到图像处理，从推荐系统到自动驾驶，AI 正在改变世界。"
}

def mock_chat_completion(messages, temperature=0.7, max_tokens=500, **kwargs):
    """模拟 OpenAI 响应"""
    user_msg = messages[-1]["content"]
    
    # 查找匹配的响应
    response_text = None
    for key, value in MOCK_RESPONSES.items():
        if key in user_msg:
            if isinstance(value, dict):
                response_text = value.get(temperature, value.get(0.5))
            else:
                response_text = value
            break
    
    if response_text is None:
        response_text = f"这是对 '{user_msg[:30]}...' 的模拟回复。"
    
    # 构造模拟响应
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    mock_response.usage.prompt_tokens = len(user_msg) // 2
    mock_response.usage.completion_tokens = len(response_text) // 2
    mock_response.usage.total_tokens = mock_response.usage.prompt_tokens + mock_response.usage.completion_tokens
    
    return mock_response

# ============ 使用 Mock 运行 ============
def main():
    print("=" * 50)
    print("Task 1 (Mock 版本): 第一个 LLM API 调用")
    print("=" * 50)
    print("⚠️ 当前使用 Mock 数据，设置 DEEPSEEK_API_KEY 后可调用真实 API\n")
    
    # 基础调用
    print("--- 基础调用 ---")
    response = mock_chat_completion([
        {"role": "user", "content": "你好，请用一句话介绍你自己"}
    ])
    print(f"回复: {response.choices[0].message.content}")
    
    # 对比 temperature
    print("\n--- Temperature 对比 ---")
    prompt = "用一句话描述春天"
    
    for temp in [0.1, 0.5, 0.9]:
        response = mock_chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=temp
        )
        print(f"temperature={temp}: {response.choices[0].message.content}")
    
    # Token 用量统计
    print("\n--- Token 用量 ---")
    response = mock_chat_completion(
        [{"role": "user", "content": "写一篇 500 字的文章，介绍人工智能的发展历史"}],
        max_tokens=600
    )
    usage = response.usage
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    print(f"预估成本: ¥{usage.prompt_tokens * 0.000001 + usage.completion_tokens * 0.000002:.4f}")
    
    print("\n" + "=" * 50)
    print("✅ Mock 版本运行成功！")
    print("💡 设置 DEEPSEEK_API_KEY 后，将代码中的 mock_chat_completion 替换为真实 API 调用")
    print("=" * 50)

if __name__ == "__main__":
    main()
