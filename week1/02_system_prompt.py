"""
Task 2: System Prompt 设计 + 角色扮演
目标：理解 System Prompt 对模型行为的控制作用
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def chat(messages: list[dict], temperature: float = 0.7) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=temperature,
        max_tokens=500
    )
    return response.choices[0].message.content

def main():
    print("=" * 50)
    print("Task 2: System Prompt 设计")
    print("=" * 50)
    
    user_msg = "帮我写一段关于人工智能的文案"
    
    # 1. 无 System Prompt
    print("\n--- 1. 无 System Prompt ---")
    response = chat([{"role": "user", "content": user_msg}])
    print(response)
    
    # 2. 有 System Prompt
    print("\n--- 2. 有 System Prompt ---")
    system_prompt = """你是一位专业的科技自媒体作者，擅长用通俗易懂的语言解释技术概念。
要求：
1. 字数控制在 150 字以内
2. 使用比喻或类比
3. 避免专业术语
4. 语气亲切自然"""
    response = chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ])
    print(response)
    
    # 3. 角色扮演对比
    print("\n--- 3. 角色扮演对比 ---")
    
    # 严肃版
    serious_prompt = "你是一位严肃的大学教授，回答问题严谨、专业，使用学术语言。"
    response = chat([
        {"role": "system", "content": serious_prompt},
        {"role": "user", "content": "什么是机器学习？"}
    ])
    print(f"严肃版: {response}\n")
    
    # 活泼版
    fun_prompt = "你是一个活泼可爱的卡通人物，说话活泼，经常用表情符号和口语化表达。"
    response = chat([
        {"role": "system", "content": fun_prompt},
        {"role": "user", "content": "什么是机器学习？"}
    ])
    print(f"活泼版: {response}")

if __name__ == "__main__":
    main()
