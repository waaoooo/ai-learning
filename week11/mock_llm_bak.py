import json
class MockLLM:
    """模拟 LLM，返回预设响应"""
    def __init__(self, preset_responses: dict):
        """
        preset_responses: {消息关键词: 响应内容}
        例如: {"天气": "北京今天晴，25°C"}
        """
        self.preset = preset_responses
        self.call_count = 0
        self.call_history = []
    def chat(self, messages: list) -> str:
        self.call_count += 1
        user_msg = messages[-1]["content"] if messages else ""
        self.call_history.append(user_msg)
        # 匹配预设响应
        for keyword, response in self.preset.items():
            if keyword in user_msg:
                return response
        return "抱歉，我不知道"
    def reset(self):
        self.call_count = 0
        self.call_history.clear()
# === 验收测试 ===
if __name__ == "__main__":
    mock = MockLLM({
        "天气": "北京今天晴，25°C",
        "时间": "现在是下午 3 点",
    })
    # 测试 1：命中预设
    r1 = mock.chat([{"role": "user", "content": "今天天气怎么样？"}])
    print(f"测试1: {r1}")  # 期望: 北京今天晴，25°C
    # 测试 2：未命中
    r2 = mock.chat([{"role": "user", "content": "你好"}])
    print(f"测试2: {r2}")  # 期望: 抱歉，我不知道
    # 测试 3：调用计数
    print(f"调用次数: {mock.call_count}")  # 期望: 2
    print(f"调用历史: {mock.call_history}")  # 期望: ['今天天气怎么样？', '你好']
