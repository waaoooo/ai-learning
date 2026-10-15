"""
Task 3: 结构化输出（JSON Mode + Function Calling）
目标：让 LLM 输出严格的 JSON 格式，用于程序解析
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def extract_info_json_mode(text: str) -> dict:
    """方法1：JSON Mode（推荐，简单场景）"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个信息提取助手，请严格按照 JSON 格式输出。"},
            {"role": "user", "content": f"从以下文本中提取信息：\n{text}"}
        ],
        response_format={"type": "json_object"},  # 关键：启用 JSON Mode
        temperature=0.1
    )
    return json.loads(response.choices[0].message.content)

def extract_info_function_calling(text: str) -> dict:
    """方法2：Function Calling（推荐，复杂场景）"""
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
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "技能列表"
                    },
                    "is_student": {"type": "boolean", "description": "是否学生"}
                },
                "required": ["name", "age"]
            }
        }
    }]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": f"提取信息：\n{text}"}
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "extract_person_info"}},
        temperature=0.1
    )
    
    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)

def main():
    print("=" * 50)
    print("Task 3: 结构化输出")
    print("=" * 50)
    
    text = "张三，25 岁，会 Python 和 JavaScript，目前还在上大学。"
    
    # 方法1：JSON Mode
    print("\n--- 方法1: JSON Mode ---")
    try:
        result = extract_info_json_mode(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"JSON Mode 失败: {e}")
    
    # 方法2：Function Calling
    print("\n--- 方法2: Function Calling ---")
    try:
        result = extract_info_function_calling(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Function Calling 失败: {e}")
    
    # 对比总结
    print("\n--- 对比总结 ---")
    print("JSON Mode:")
    print("  - 优点：简单，不需要定义 Schema")
    print("  - 缺点：无法控制字段类型，可能输出多余字段")
    print("  - 适用：简单提取、快速原型")
    print("\nFunction Calling:")
    print("  - 优点：精确控制字段类型，支持复杂嵌套")
    print("  - 缺点：需要定义工具描述，token 消耗更多")
    print("  - 适用：生产环境、复杂数据结构")

if __name__ == "__main__":
    main()
