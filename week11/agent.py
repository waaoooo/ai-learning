class SimpleAgent:
    def __init__(self, llm):
        self.llm = llm

    def ask(self, question: str) -> str:
        if not question or not question.strip():
            return "请输入有效问题"
        response = self.llm.chat([{"role": "user", "content": question}])
        return response


class ToolAgent:
    def __init__(self, llm, tools: dict):
        self.llm = llm
        self.tools = tools  # {"tool_name": callable}

    def run(self, question: str) -> str:
        # Step 1: LLM 决定用哪个工具
        decision = self.llm.chat([
            {"role": "system", "content": f"可用工具: {list(self.tools.keys())}"},
            {"role": "user", "content": question},
        ])
        # decision 格式: "use:weather:北京"
        if decision.startswith("use:"):
            parts = decision.split(":")
            tool_name = parts[1]
            arg = parts[2] if len(parts) > 2 else ""
            if tool_name in self.tools:
              try:
                tool_result = self.tools[tool_name](arg)
              except Exception as e:
                tool_result = f"工具执行失败：{e}"
                # Step 2: LLM 根据工具结果回答
              final = self.llm.chat([
                   {"role": "user", "content": f"问题: {question}\n工具结果: {tool_result}\n请回答"},
              ])
              return final
            else:
              return "抱歉，我不知道"
        return decision

