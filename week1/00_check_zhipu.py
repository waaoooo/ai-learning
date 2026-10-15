"""
第 2 步：探测你的智谱 Key 能用哪些模型
用法：填好 .env 后运行  python3 00_check_zhipu.py
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# 候选模型：免费的排前面，省成本
CANDIDATES = [
    "glm-4.7-flash",
    "glm-4.5-flash",
    "glm-4-flash",
    "glm-4.6",
    "glm-4.5",
    "glm-4.5-air",
]

if not API_KEY or "填这里" in API_KEY:
    print("❌ 还没填 ZHIPU_API_KEY，请先编辑 .env")
    raise SystemExit(1)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print("=" * 50)
print("1) 列出官方 /models 接口返回的模型")
print("=" * 50)
try:
    models = client.models.list()
    for m in models.data:
        print("  -", m.id)
except Exception as e:
    print(f"  （/models 接口不可用：{e}）")

print("\n" + "=" * 50)
print("2) 逐个试候选模型，找出能用的")
print("=" * 50)
working = []
for name in CANDIDATES:
    try:
        r = client.chat.completions.create(
            model=name,
            messages=[{"role": "user", "content": "只回复两个字：可用"}],
            temperature=0.1,
            max_tokens=20,
        )
        reply = r.choices[0].message.content.strip()
        print(f"  ✅ {name:20s} -> {reply}")
        working.append(name)
    except Exception as e:
        msg = str(e)[:80]
        print(f"  ❌ {name:20s} -> {msg}")

print("\n" + "=" * 50)
if working:
    print(f"✅ 可用模型：{working}")
    print(f"💡 建议下一步用第一个（免费优先）：{working[0]}")
else:
    print("❌ 没有模型可用，检查 Key 或账户额度")
print("=" * 50)
