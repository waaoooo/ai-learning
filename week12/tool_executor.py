# tool_executor.py
import json
from typing import Callable, Dict, Any

class ToolExecutor:
    """工具执行器：注册 + 校验 + 执行 + 错误处理"""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: Dict[str, dict] = {}

    def register(self, name: str, func: Callable, schema: dict):
        """注册工具"""
        self.tools[name] = func
        self.schemas[name] = schema

    def execute(self, name: str, **kwargs) -> dict:
        """执行工具，返回 {result, error}"""
        if name not in self.tools:
            return {"result": None, "error": f"未知工具: {name}"}

        # 参数校验
        schema = self.schemas[name]
        required = schema.get("required", [])
        for param in required:
            if param not in kwargs:
                return {"result": None, "error": f"缺少参数: {param}"}

        # 执行
        try:
            result = self.tools[name](**kwargs)
            return {"result": result, "error": None}
        except Exception as e:
            return {"result": None, "error": f"执行失败: {e}"}

    def list_tools(self) -> list:
        return list(self.tools.keys())

    def get_schema(self, name: str) -> dict:
        return self.schemas.get(name, {})


# === 注册工具 ===
def weather_tool(city: str) -> str:
    return f"{city}今天晴，25°C"

def search_tool(query: str, top_k: int = 3) -> list:
    return [f"结果{i}: {query} 相关内容" for i in range(top_k)]

if __name__ == "__main__":
    executor = ToolExecutor()
    executor.register("weather", weather_tool, {"required": ["city"]})
    executor.register("search", search_tool, {"required": ["query"]})

    print(f"已注册工具: {executor.list_tools()}")

    # 正常调用
    r1 = executor.execute("weather", city="北京")
    print(f"天气: {r1}")

    # 缺少参数
    r2 = executor.execute("weather")
    print(f"缺少参数: {r2}")

    # 未知工具
    r3 = executor.execute("unknown_tool", x=1)
    print(f"未知工具: {r3}")

    # 工具异常
    def broken_tool(x: str) -> str:
        raise ValueError("模拟异常")
    executor.register("broken", broken_tool, {"required": ["x"]})
    r4 = executor.execute("broken", x="test")
    print(f"工具异常: {r4}")
