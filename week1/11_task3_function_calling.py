"""
Task 3（第 2 步）：Function Calling —— 精确控制字段和类型
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

USER_MSG = "提取这句话里的信息：张三，25 岁，会 Python 和 JavaScript，目前还在上大学。"

# 定义工具（相当于给模型一个"填空模板"）
tools = [{
    "type": "function",
    "function": {
        "name": "extract_person_info",
        "description": "从文本中提取人物信息",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "姓名"},
                "age": {"type": "integer", "description": "年龄，必须是整数"},
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "技能列表",
                },
                "is_student": {"type": "boolean", "description": "是否在校学生"},
            },
            "required": ["name", "age", "skills", "is_student"],
        },
    },
}]

r = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[{"role": "user", "content": USER_MSG}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_person_info"}},
    temperature=0.1,
    extra_body={"thinking": {"type": "disabled"}},
)

# 从 tool_call 里取参数
tool_call = r.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)

print("=" * 60)
print("Function Calling 结果")
print("=" * 60)
print(f"模型选择的工具: {tool_call.function.name}")
print(f"参数: {json.dumps(args, ensure_ascii=False, indent=2)}")

# 类型校验
print("\n--- 类型校验 ---")
checks = {
    "name 是字符串": isinstance(args["name"], str),
    "age 是整数": isinstance(args["age"], int),
    "skills 是列表": isinstance(args["skills"], list),
    "is_student 是布尔": isinstance(args["is_student"], bool),
}
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")
print(f"\nage 的值: {args['age']} (类型: {type(args['age']).__name__})")
