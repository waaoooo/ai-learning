"""
W2 第 3 步：Embedding —— 语义相似度
把 3 句话变成向量，算两两余弦相似度
"""
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

SENTENCES = [
    "一线城市住宿标准不超过500元",      # 主题：住宿报销
    "出差住酒店的报销限额是多少",        # 主题：住宿报销（换了个说法）
    "公司年假一共有几天",               # 主题：年假（无关）
]


def embed(texts):
    for attempt in range(3):
        try:
            r = client.embeddings.create(model="embedding-3", input=texts)
            return [d.embedding for d in r.data]
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  (限流，重试)")
            time.sleep(2 ** attempt)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)


print("=" * 60)
print("正在获取 embedding 向量 ...")
print("=" * 60)
vecs = embed(SENTENCES)
print(f"每句话被转成 {len(vecs[0])} 维的向量\n")

print("=" * 60)
print("两两余弦相似度（越接近 1 越相似）")
print("=" * 60)
for i in range(len(SENTENCES)):
    for j in range(i + 1, len(SENTENCES)):
        sim = cosine(vecs[i], vecs[j])
        print(f"  [{i}] vs [{j}]  {sim:.3f}")
        print(f"      {SENTENCES[i]}")
        print(f"      {SENTENCES[j]}")
        print()

print("结论：语义相近的句子，向量距离也近 —— 这就是检索的基础。")
