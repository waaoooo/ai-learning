# -*- coding: utf-8 -*-
"""L2-01 Hello RAG: 最小可运行的 RAG 示例（GLM + faiss）"""
import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../../week1/.env")

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

docs = [
    "公司年假：入职满1年享有5天年假，满3年10天，满5年15天。",
    "报销流程：员工提交发票→直属经理审批→财务复核→3个工作日到账。",
    "加班补偿：工作日加班按1.5倍，周末2倍，法定节假日3倍。",
]


import hashlib

def _local_embed(texts, dim=256):
    """离线 fallback：基于字符 n-gram 哈希的伪向量（仅用于演示管线，非生产）。"""
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
        if nrm > 0:
            v /= nrm
        out.append(v)
    return np.array(out, dtype="float32")


def embed(texts):
    try:
        r = client.embeddings.create(model="embedding-3", input=texts)
        return np.array([d.embedding for d in r.data], dtype="float32")
    except Exception as e:
        print(f"[warn] 真实 embedding 不可用（{e}），使用本地 fallback embedding 演示。")
        return _local_embed(texts)


def main():
    V = embed(docs)
    index = faiss.IndexFlatIP(V.shape[1])
    faiss.normalize_L2(V)
    index.add(V)

    q = "年假怎么算？"
    qv = embed([q])
    faiss.normalize_L2(qv)
    D, I = index.search(qv, 2)
    ctx = "\n".join(docs[i] for i in I[0])
    print("检索到：", [docs[i] for i in I[0]])

    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": "基于提供的资料回答，资料不足请说不知道。用中文简洁回答。"},
            {"role": "user", "content": f"资料：\n{ctx}\n\n问题：{q}"},
        ],
    )
    print("答案：", resp.choices[0].message.content)


if __name__ == "__main__":
    main()
