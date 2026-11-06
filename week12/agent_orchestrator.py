# agent_orchestrator.py
import json
import re
from llm_client import LLMClient
from tool_executor import ToolExecutor
from rag_engine import SimpleRAGEngine

class AgentOrchestrator:
    """Agent 编排器：路由 + 调度 + 上下文管理"""

    def __init__(self, llm: LLMClient, tools: ToolExecutor, rag: SimpleRAGEngine):
        self.llm = llm
        self.tools = tools
        self.rag = rag
        self.context = []  # 短期记忆

    def route(self, question: str) -> str:
        """改进的路由：严格 prompt + 解析 + fallback"""
        prompt = f"""你是一个路由分类器。根据用户问题，只回复以下之一：
    - rag（需要查询知识库）
    - tool:weather:城市名（需要查询天气）
    - tool:search:关键词（需要搜索）
    - direct（直接回答）
    
    只回复分类结果，不要解释。用户问题: {question}"""
        decision = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
    
        # 解析：提取关键词
        decision = decision.strip().lower()
        if "rag" in decision:
            return "rag"
        match = re.search(r"tool:(\w+):(.+)", decision)
        if match:
            return f"tool:{match.group(1)}:{match.group(2).strip()}"
        return "direct"

    def handle(self, question: str) -> str:
        """处理用户问题"""
        self.context.append({"role": "user", "content": question})

        decision = self.route(question)
        print(f"  路由决策: {decision}")

        if decision == "rag":
            answer = self.rag.answer(question)
        elif decision.startswith("tool:"):
            parts = decision.split(":")
            tool_name = parts[1]
            arg = parts[2] if len(parts) > 2 else ""
            result = self.tools.execute(tool_name, **{self._infer_param(tool_name): arg})
            if result["error"]:
                answer = f"工具调用失败: {result['error']}"
            else:
                answer = self.llm.chat([
                    {"role": "user", "content": f"问题: {question}\n工具结果: {result['result']}\n请回答"}
                ])
        else:
            answer = self.llm.chat(self.context)

        self.context.append({"role": "assistant", "content": answer})
        return answer

    def _infer_param(self, tool_name: str) -> str:
        """推断工具参数名"""
        schemas = {
            "weather": "city",
            "search": "query",
        }
        return schemas.get(tool_name, "input")

    def reset_context(self):
        self.context = []

if __name__ == "__main__":
    llm = LLMClient()
    tools = ToolExecutor()
    tools.register("weather", lambda city: f"{city}今天晴，25°C", {"required": ["city"]})
    tools.register("search", lambda query: f"搜索结果: {query}", {"required": ["query"]})

    rag = SimpleRAGEngine(llm)
    rag.add_documents(["RAG 是检索增强生成技术。\n\nRAG 的优势是利用外部知识。"])

    orch = AgentOrchestrator(llm, tools, rag)

    questions = [
        "什么是 RAG？",
        "北京天气怎么样？",
        "你好",
    ]

    for q in questions:
        print(f"\n用户: {q}")
        answer = orch.handle(q)
        print(f"Agent: {answer[:100]}...")
