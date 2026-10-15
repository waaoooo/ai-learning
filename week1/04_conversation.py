"""
Task 4: 多轮对话管理
目标：实现对话历史管理，理解上下文窗口和 token 管理
"""
import os
from collections import deque
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

class ConversationManager:
    """多轮对话管理器"""
    
    def __init__(self, system_prompt: str, max_history: int = 10, max_tokens: int = 4000):
        self.system_prompt = system_prompt
        self.max_history = max_history  # 保留最近 N 轮对话
        self.max_tokens = max_tokens  # 上下文窗口大小
        self.history = deque(maxlen=max_history * 2)  # 每轮 = user + assistant
        self.total_tokens = 0
    
    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数（中文约 1.5 字符/token，英文约 4 字符/token）"""
        return len(text) // 2
    
    def chat(self, user_message: str) -> str:
        # 构建消息列表
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})
        
        # 调用 API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_message = response.choices[0].message.content
        usage = response.usage
        
        # 更新统计
        self.total_tokens += usage.total_tokens
        
        # 更新历史
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def clear(self):
        """清空对话历史"""
        self.history.clear()
        self.total_tokens = 0
    
    def get_history(self) -> list:
        """获取当前对话历史"""
        return list(self.history)
    
    def get_stats(self) -> dict:
        """获取对话统计"""
        return {
            "history_count": len(self.history),
            "total_tokens": self.total_tokens,
            "estimated_cost": self.total_tokens * 0.000001  # 粗略估算
        }

def main():
    print("=" * 50)
    print("Task 4: 多轮对话管理")
    print("=" * 50)
    
    conv = ConversationManager(
        system_prompt="你是一个友好的技术助手，回答简洁明了。",
        max_history=5
    )
    
    # 模拟多轮对话
    questions = [
        "什么是机器学习？",
        "它和深度学习有什么关系？",
        "能给我一个实际的例子吗？",
    ]
    
    for q in questions:
        print(f"\n👤 用户: {q}")
        response = conv.chat(q)
        print(f"🤖 助手: {response}")
    
    # 显示统计
    print(f"\n--- 对话统计 ---")
    stats = conv.get_stats()
    print(f"历史记录数: {stats['history_count']}")
    print(f"总 token 数: {stats['total_tokens']}")
    print(f"预估成本: ¥{stats['estimated_cost']:.4f}")
    
    # 显示历史
    print(f"\n--- 历史记录 ---")
    for i, msg in enumerate(conv.get_history(), 1):
        print(f"  {i}. {msg['role']}: {msg['content'][:50]}...")

if __name__ == "__main__":
    main()
