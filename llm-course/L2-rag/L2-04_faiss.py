# -*- coding: utf-8 -*-
"""L2-04 faiss 实战：构建 / 检索 / 保存 / 加载"""
import os, hashlib, numpy as np, faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../../week1/.env")
client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4")

docs = [
    "公司年假：入职满1年享有5天年假，满3年10天，满5年15天。",
    "报销流程：员工提交发票→直属经理审批→财务复核→3个工作日到账。",
    "加班补偿：工作日加班按1.5倍，周末2倍，法定节假日3倍。",
    "体检福利：每年一次免费体检，入职满1年可带家属。",
]

def _local_embed(texts, dim=256):
    out = []
    for t in texts:
        v = np.zeros(dim, dtype="float32")
        for i in range(len(t)):
            for n in (1, 2, 3):
                if i + n <= len(t):
                    gram = t[i:i + n]
                    h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                    v[h % dim] += 1.0
        nrm = np.linalg.norm(v)
        if nrm > 0: v /= nrm
        out.append(v)
    return np.array(out, dtype="float32")

def embed(texts):
    try:
        r = client.embeddings.create(model="embedding-3", input=texts)
        V = np.array([d.embedding for d in r.data], dtype="float32")
        V /= np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
        return V
    except Exception as e:
        print(f"[warn] 真实 embedding 不可用（{e}），使用本地 fallback。")
        return _local_embed(texts)

def main():
    V = embed(docs)
    dim = V.shape[1]

    # 1. IndexFlatIP（归一化后内积 == cosine）
    index = faiss.IndexFlatIP(dim)
    index.add(V)
    print("索引向量数:", index.ntotal)

    # 2. 检索
    q = "年假怎么算？"
    qv = embed([q])
    D, I = index.search(qv, 2)
    print("\n--- 检索 top-2 ---")
    for d, i in zip(D[0], I[0]):
        print(f"score={d:.4f}  {docs[i]}")

    # 3. 保存
    faiss.write_index(index, "index.faiss")
    with open("docs.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(docs))
    print("\n已保存 index.faiss + docs.txt")

    # 4. 重新加载
    idx2 = faiss.read_index("index.faiss")
    docs2 = open("docs.txt", encoding="utf-8").read().split("\n")
    D2, I2 = idx2.search(qv, 2)
    print("\n--- 重新加载后检索 ---")
    for d, i in zip(D2[0], I2[0]):
        print(f"score={d:.4f}  {docs2[i]}")

    # 5. HNSW 对比（可选）
    hnsw = faiss.IndexHNSWFlat(dim, 32)
    hnsw.add(V)
    Dh, Ih = hnsw.search(qv, 2)
    print("\n--- HNSW top-2 ---")
    for d, i in zip(Dh[0], Ih[0]):
        print(f"score={d:.4f}  {docs[i]}")

if __name__ == "__main__":
    main()
