# -*- coding: utf-8 -*-
"""L2-07 RAG 诊断脚本：打印检索+答案+诊断提示"""
import os, hashlib, numpy as np, faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../../week1/.env")
client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4")

docs = [
    "公司年假：入职满1年享有5天年假，满3年10天，满5年15天。年假需提前3个工作日申请。",
    "报销流程：员工提交发票→直属经理审批→财务复核→3个工作日到账。单笔超过5000元需VP审批。",
    "加班补偿：工作日加班按1.5倍，周末2倍，法定节假日3倍。加班需提前申请并获批。",
    "体检福利：每年一次免费体检，入职满1年可带家属，家属体检半价。",
    "考勤：工作时间9:00-18:00，迟到15分钟内扣50元，超过30分钟按半天事假。",
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
    except Exception:
        return _local_embed(texts)

V = embed(docs)
index = faiss.IndexFlatIP(V.shape[1])
index.add(V)

questions = [
    "年假怎么算？",
    "公司有股票期权吗？",
    "加班费",
]

for q in questions:
    print("=" * 60)
    print("Q:", q)
    qv = embed([q])
    D, I = index.search(qv, 3)
    hits = list(zip(D[0].tolist(), I[0].tolist()))
    print("检索:")
    for score, idx in hits:
        print(f"  score={score:.3f}  {docs[idx][:50]}")
    # 诊断
    diag = []
    if hits[0][0] < 0.2:
        diag.append("低相似度：可能未召回，考虑查询改写/混合检索")
    if all(s < 0.15 for s, _ in hits):
        diag.append("无高相关结果： likely 未提取到答案")
    ctx = "\n".join(docs[i] for _, i in hits if _ >= 0.15)
    prompt = f"基于资料回答，资料不足说不知道。\n资料：\n{ctx}\n问题：{q}"
    resp = client.chat.completions.create(model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}])
    ans = resp.choices[0].message.content
    print("A:", ans)
    if len(ans or "") < 20:
        diag.append("答案过短：可能不完整")
    if diag:
        print("诊断提示:", "; ".join(diag))
    else:
        print("诊断提示: 无明显问题")

if __name__ == "__main__":
    pass
