from mock_llm import MockLLM
from agent import ToolAgent

def weather_tool(city: str) -> str:
    return f"{city}今天晴，25°C"

def test_tool_call_flow():
    mock = MockLLM({
        "可用工具": "use:weather:北京",       # 第1次调用：决定用工具
        "工具结果": "北京今天晴，25°C，适合出游",  # 第2次调用：生成回答
    })
    agent = ToolAgent(mock, {"weather": weather_tool})
    result = agent.run("北京天气？")

    assert result == "北京今天晴，25°C，适合出游"
    assert mock.call_count == 2
    print(f"✅ 工具调用流程测试通过 (调用 {mock.call_count} 次)")

def test_unknown_tool():
    mock = MockLLM({"可用工具": "use:calculator:1+1"})
    agent = ToolAgent(mock, {"weather": weather_tool})
    result = agent.run("算一下")
    assert "不知道" in result or result == "抱歉，我不知道"
    print("✅ 未知工具测试通过")

if __name__ == "__main__":
    test_tool_call_flow()
    test_unknown_tool()
    print("\n🎉 集成测试全部通过")

