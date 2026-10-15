"""
统一 Mock 模块：无 API Key 时模拟 LLM 响应
设置环境变量 USE_MOCK=true 启用
"""
import os
import time
from unittest.mock import MagicMock

MOCK_RESPONSES = {
    # Task 2: System Prompt
    "帮我写一段关于人工智能的文案": {
        "default": "人工智能是当今科技领域最热门的话题之一，它正在改变着我们的生活方式。",
        "serious": "人工智能（Artificial Intelligence）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "fun": "哇！AI 就像是现代版的魔法，让机器也能'思考'啦！🤖✨ 它可以让电脑变得更聪明，帮我们做很多事情哦~"
    },
    
    # Task 3: 结构化输出
    "提取信息": {
        "default": '{"name": "张三", "age": 25, "skills": ["Python", "JavaScript"], "is_student": true}'
    },
    
    # Task 4: 多轮对话
    "什么是机器学习？": "机器学习是人工智能的一个分支，它让计算机能够从数据中学习规律，而无需明确编程。",
    "它和深度学习有什么关系？": "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习方式。",
    "能给我一个实际的例子吗？": "比如，当你使用淘宝推荐商品时，背后就是机器学习算法在分析你的浏览和购买历史。",
    
    # Task 5: 客服
    "你好": "您好！我是小智，很高兴为您服务！请问有什么可以帮您？😊",
    "订单什么时候发货": "一般情况下，订单会在 24 小时内发货，您可以在订单详情中查看物流信息。",
    "退换货": "我们支持 7 天无理由退换货，商品需保持原包装完好。您可以在'我的订单'中申请售后。"
}

def mock_chat_completion(messages, temperature=0.7, max_tokens=500, **kwargs):
    """模拟 OpenAI 响应"""
    user_msg = messages[-1]["content"]
    system_msg = messages[0]["content"] if messages[0]["role"] == "system" else ""
    
    # 查找匹配的响应
    response_text = None
    
    # 特殊处理：结构化输出
    if "json_object" in str(kwargs.get("response_format", "")) or "function" in str(kwargs.get("tools", "")):
        response_text = '{"name": "张三", "age": 25, "skills": ["Python", "JavaScript"], "is_student": true}'
    elif "严肃" in system_msg or "教授" in system_msg:
        response_text = MOCK_RESPONSES.get("帮我写一段关于人工智能的文案", {}).get("serious", "这是严肃版回复。")
    elif "活泼" in system_msg or "卡通" in system_msg:
        response_text = MOCK_RESPONSES.get("帮我写一段关于人工智能的文案", {}).get("fun", "这是活泼版回复。")
    else:
        # 模糊匹配
        for key, value in MOCK_RESPONSES.items():
            if isinstance(value, dict):
                continue
            if key in user_msg or user_msg in key:
                response_text = value
                break
    
    if response_text is None:
        response_text = f"这是对 '{user_msg[:30]}' 的模拟回复。设置 DEEPSEE_API_KEY 后可获取真实响应。"
    
    # 构造模拟响应
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    mock_response.choices[0].message.tool_calls = None
    
    # 模拟 tool_calls（用于 Function Calling）
    if kwargs.get("tools"):
        mock_response.choices[0].message.tool_calls = [MagicMock()]
        mock_response.choices[0].message.tool_calls[0].function.arguments = response_text
    
    # Token 统计
    mock_response.usage.prompt_tokens = len(str(messages)) // 2
    mock_response.usage.completion_tokens = len(response_text) // 2
    mock_response.usage.total_tokens = mock_response.usage.prompt_tokens + mock_response.usage.completion_tokens
    
    # 模拟网络延迟
    time.sleep(0.1)
    
    return mock_response

def get_client():
    """获取客户端（真实或 Mock）"""
    if os.getenv("DEEPSEEK_API_KEY"):
        from openai import OpenAI
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        ), False
    else:
        return None, True  # None 表示使用 mock
