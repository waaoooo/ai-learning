"""
探测：推理模型的 3 种应对方式，哪种能用
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

def probe(tag, model, extra=None, max_tokens=300):
    print("=" * 60)
    print(f"{tag}  model={model}  extra={extra}  max_tokens={max_tokens}")
    print("=" * 60)
    try:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": "用一句话描述春天"}],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        if extra:
            kwargs["extra_body"] = extra
        r = client.chat.completions.create(**kwargs)
        msg = r.choices[0].message
        content = msg.content
        reasoning = getattr(msg, "reasoning_content", None)
        print(f"  content        : {content!r}")
        print(f"  reasoning      : {(reasoning[:60] + '...') if reasoning else None!r}")
        print(f"  usage          : {r.usage}")
    except Exception as e:
        print(f"  ❌ 报错: {str(e)[:200]}")
    print()

probe("A) 推理模型 + 关闭思考", "glm-4.7-flash",
      extra={"thinking": {"type": "disabled"}})

probe("B) 推理模型 + 开思考 + 大额度", "glm-4.7-flash",
      extra={"thinking": {"type": "enabled"}}, max_tokens=2000)

probe("C) 非推理模型（对照组）", "glm-4-flash")
