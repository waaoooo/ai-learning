from mock_llm import MockLLM
from agent import SimpleAgent

def test_empty_question():
    mock = MockLLM({})
    agent = SimpleAgent(mock)
    assert agent.ask("") == "请输入有效问题"
    assert agent.ask("   ") == "请输入有效问题"
    print("✅ 空问题测试通过")

def test_normal_question():
    mock = MockLLM({"天气": "北京晴，25°C"})
    agent = SimpleAgent(mock)
    result = agent.ask("今天天气？")
    assert result == "北京晴，25°C"
    assert mock.call_count == 1
    print("✅ 正常问题测试通过")

def test_unknown_question():
    mock = MockLLM({"天气": "北京晴，25°C"})
    agent = SimpleAgent(mock)
    result = agent.ask("你好")
    assert result == "抱歉，我不知道"
    print("✅ 未知问题测试通过")

if __name__ == "__main__":
    test_empty_question()
    test_normal_question()
    test_unknown_question()
    print("\n🎉 全部测试通过")

