class MockLLM:
    def __init__(self, preset_responses: dict):
        self.preset = preset_responses
        self.call_count = 0
        self.call_history = []

    def chat(self, messages: list) -> str:
        self.call_count += 1
        # 搜索所有消息，不只是最后一条
        full_text = " ".join(m.get("content", "") for m in messages)
        self.call_history.append(full_text)

        for keyword, response in self.preset.items():
            if keyword in full_text:
                return response
        return "抱歉，我不知道"

    def reset(self):
        self.call_count = 0
        self.call_history.clear()

