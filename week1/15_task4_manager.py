"""
Task 4（第 3 步）：ConversationManager —— 滑动窗口 + 记忆注入
可复用到后续所有任务
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)


class ConversationManager:
    def __init__(self, system_prompt: str, max_turns: int = 5, memory: str = ""):
        self.system_prompt = system_prompt
        self.max_turns = max_turns      # 保留最近 N 轮
        self.memory = memory            # 长期记忆，每次都注入
        self.history = []               # 最近 N 轮对话

    def _build_messages(self, user_msg: str) -> list:
        system = self.system_prompt
        if self.memory:
            system += f"\n\n【用户记忆】\n{self.memory}"
        return (
            [{"role": "system", "content": system}]
            + self.history[-(self.max_turns * 2):]
            + [{"role": "user", "content": user_msg}]
        )

    def chat(self, user_msg: str) -> str:
        r = client.chat.completions.create(
            model="glm-4.7-flash",
            messages=self._build_messages(user_msg),
            temperature=0.5,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )
        answer = r.choices[0].message.content
        # 更新历史（只保留最近 N 轮）
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": answer})
        self.history = self.history[-(self.max_turns * 2):]
        self.last_usage = r.usage
        return answer

    def clear(self):
        self.history = []


if __name__ == "__main__":
    # 演示：即使对话超过 5 轮，用户偏好（记忆）始终生效
    conv = ConversationManager(
        system_prompt="你是一个助手，回答简洁。",
        max_turns=3,
        memory="用户是一名后端开发者，偏好 Python，讨厌 Java。",
    )

    print("=" * 60)
    print("第 1 轮：问技术偏好")
    print("=" * 60)
    print("用户: 我该学什么语言？")
    print("助手:", conv.chat("我该学什么语言？"))

    print("\n" + "=" * 60)
    print("第 2-4 轮：聊别的（把窗口占满）")
    print("=" * 60)
    for topic in ["今天天气", "推荐一本书", "讲个笑话"]:
        print(f"用户: {topic}")
        print("助手:", conv.chat(topic))

    print("\n" + "=" * 60)
    print("第 5 轮：再问偏好 —— 记忆还在吗？")
    print("=" * 60)
    print("用户: 我最讨厌什么语言？")
    print("助手:", conv.chat("我最讨厌什么语言？"))
    print(f"\n[当前窗口内消息数: {len(conv.history)}]")
