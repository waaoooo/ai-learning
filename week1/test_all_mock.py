"""
Week 1 完整测试：运行所有 Task 的 Mock 版本
"""
import os
import sys
import json
from unittest.mock import MagicMock

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))
from mock_llm import mock_chat_completion

def test_task2():
    """Task 2: System Prompt 设计"""
    print("\n" + "=" * 50)
    print("Task 2: System Prompt 设计")
    print("=" * 50)
    
    user_msg = "帮我写一段关于人工智能的文案"
    
    # 1. 无 System Prompt
    print("\n--- 1. 无 System Prompt ---")
    response = mock_chat_completion([
        {"role": "user", "content": user_msg}
    ])
    print(response.choices[0].message.content)
    
    # 2. 有 System Prompt
    print("\n--- 2. 有 System Prompt ---")
    system_prompt = """你是一位专业的科技自媒体作者，擅长用通俗易懂的语言解释技术概念。
要求：
1. 字数控制在 150 字以内
2. 使用比喻或类比
3. 避免专业术语
4. 语气亲切自然"""
    response = mock_chat_completion([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ])
    print(response.choices[0].message.content)
    
    # 3. 角色扮演对比
    print("\n--- 3. 角色扮演对比 ---")
    
    serious_prompt = "你是一位严肃的大学教授，回答问题严谨、专业，使用学术语言。"
    response = mock_chat_completion([
        {"role": "system", "content": serious_prompt},
        {"role": "user", "content": "什么是机器学习？"}
    ])
    print(f"严肃版: {response.choices[0].message.content}\n")
    
    fun_prompt = "你是一个活泼可爱的卡通人物，说话活泼，经常用表情符号和口语化表达。"
    response = mock_chat_completion([
        {"role": "system", "content": fun_prompt},
        {"role": "user", "content": "什么是机器学习？"}
    ])
    print(f"活泼版: {response.choices[0].message.content}")

def test_task3():
    """Task 3: 结构化输出"""
    print("\n" + "=" * 50)
    print("Task 3: 结构化输出")
    print("=" * 50)
    
    text = "张三，25 岁，会 Python 和 JavaScript，目前还在上大学。"
    
    # 方法1：JSON Mode
    print("\n--- 方法1: JSON Mode ---")
    try:
        response = mock_chat_completion(
            [
                {"role": "system", "content": "你是一个信息提取助手，请严格按照 JSON 格式输出。"},
                {"role": "user", "content": f"从以下文本中提取信息：\n{text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        result = json.loads(response.choices[0].message.content)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"JSON Mode 失败: {e}")
    
    # 方法2：Function Calling
    print("\n--- 方法2: Function Calling ---")
    try:
        tools = [{
            "type": "function",
            "function": {
                "name": "extract_person_info",
                "description": "提取人物信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "姓名"},
                        "age": {"type": "integer", "description": "年龄"},
                        "skills": {"type": "array", "items": {"type": "string"}, "description": "技能列表"},
                        "is_student": {"type": "boolean", "description": "是否学生"}
                    },
                    "required": ["name", "age"]
                }
            }
        }]
        
        response = mock_chat_completion(
            [{"role": "user", "content": f"提取信息：\n{text}"}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "extract_person_info"}},
            temperature=0.1
        )
        result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Function Calling 失败: {e}")

def test_task4():
    """Task 4: 多轮对话管理"""
    print("\n" + "=" * 50)
    print("Task 4: 多轮对话管理")
    print("=" * 50)
    
    from collections import deque
    
    class ConversationManager:
        def __init__(self, system_prompt: str, max_history: int = 10):
            self.system_prompt = system_prompt
            self.max_history = max_history
            self.history = deque(maxlen=max_history * 2)
        
        def chat(self, user_message: str) -> str:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.history)
            messages.append({"role": "user", "content": user_message})
            
            response = mock_chat_completion(messages, temperature=0.7)
            assistant_message = response.choices[0].message.content
            
            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        
        def get_history(self) -> list:
            return list(self.history)
    
    conv = ConversationManager(
        system_prompt="你是一个友好的技术助手，回答简洁明了。",
        max_history=5
    )
    
    questions = [
        "什么是机器学习？",
        "它和深度学习有什么关系？",
        "能给我一个实际的例子吗？",
    ]
    
    for q in questions:
        print(f"\n👤 用户: {q}")
        response = conv.chat(q)
        print(f"🤖 助手: {response}")
    
    print(f"\n--- 历史记录 ({len(conv.get_history())} 条) ---")
    for i, msg in enumerate(conv.get_history(), 1):
        print(f"  {i}. {msg['role']}: {msg['content'][:50]}...")

def test_task5():
    """Task 5: 智能客服 API（模拟）"""
    print("\n" + "=" * 50)
    print("Task 5: 智能客服 API（模拟）")
    print("=" * 50)
    
    CUSTOMER_SERVICE_PROMPT = """你是一个专业的电商客服助手，名叫"小智"。
职责：
1. 解答用户关于订单、物流、退换货、产品咨询等问题
2. 态度友好、耐心，使用礼貌用语
3. 对于无法解决的问题，引导用户联系人工客服
4. 每次回复控制在 100 字以内
特殊标记：
- 当问题超出能力范围时，在回复前加上 [需要: 人工]"""
    
    conversations = {}
    
    def chat(session_id: str, user_message: str):
        if session_id not in conversations:
            conversations[session_id] = []
        
        history = conversations[session_id]
        
        messages = [{"role": "system", "content": CUSTOMER_SERVICE_PROMPT}]
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": user_message})
        
        response = mock_chat_completion(messages, temperature=0.5, max_tokens=300)
        reply = response.choices[0].message.content
        
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})
        
        need_human = "[需要: 人工]" in reply
        reply = reply.replace("[需要: 人工]", "").strip()
        
        return {"reply": reply, "need_human": need_human}
    
    # 模拟对话
    test_cases = [
        ("user_001", "你好"),
        ("user_001", "订单什么时候发货"),
        ("user_001", "那大概几天能到？"),
        ("user_002", "退换货"),
    ]
    
    for session_id, message in test_cases:
        print(f"\n👤 [{session_id}]: {message}")
        result = chat(session_id, message)
        print(f"🤖 客服: {result['reply']}")
        if result['need_human']:
            print("⚠️ 需要人工介入")
    
    print(f"\n--- 活跃会话数: {len(conversations)} ---")

def main():
    print("=" * 50)
    print("Week 1 完整测试（Mock 版本）")
    print("=" * 50)
    print("⚠️ 使用 Mock 数据，设置 DEEPSEEK_API_KEY 后可运行真实版本\n")
    
    test_task2()
    test_task3()
    test_task4()
    test_task5()
    
    print("\n" + "=" * 50)
    print("✅ Week 1 所有任务 Mock 测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()
