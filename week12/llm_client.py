# llm_client.py
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

class LLMClient:
    """LLM 客户端封装：重试 + 超时 + Token 统计"""

    def __init__(self, model: str = "glm-4-flash", max_retries: int = 3):
        self.client = OpenAI(
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            timeout=30,
        )
        self.model = model
        self.max_retries = max_retries
        self.total_tokens = 0
        self.total_calls = 0

    def chat(self, messages: list, temperature: float = 0.7) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = response.choices[0].message.content
                tokens = response.usage.total_tokens if response.usage else 0
                self.total_tokens += tokens
                self.total_calls += 1
                return content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise RuntimeError(f"LLM 调用失败 ({self.max_retries} 次): {e}")

    def stats(self) -> dict:
        return {"calls": self.total_calls, "tokens": self.total_tokens}

if __name__ == "__main__":
    llm = LLMClient()
    reply = llm.chat([{"role": "user", "content": "用一句话介绍 RAG"}])
    print(f"回复: {reply}")
    print(f"统计: {llm.stats()}")
