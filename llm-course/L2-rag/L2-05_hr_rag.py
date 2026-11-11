# -*- coding: utf-8 -*-
"""L2-05 HR 制度智能问答：端到端 RAG"""
import os, hashlib, numpy as np, faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../../week1/.env")
client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4")

# HR 制度库（每条为一个 chunk，带制度名 metadata）
docs = [
    ("年假制度", "公司年假：入职满1年享有5天年假，满3年10天，满5年15天。年假需提前3个工作日申请。"),
    ("报销制度", "报销流程：员工提交发票→直属经理审批→财务复核→3个工作日到账。单笔超过5000元需VP审批。"),
    ("加班制度", "加班补偿：工作日加班按1.5倍，周末2倍，法定节假日3倍。加班需提前申请并获批。"),
    ("体检福利", "体检福利：每年一次免费体检，入职满1年可带家属，家属体检半价。"),
    ("考勤制度", "考勤：工作时间9:00-18:00，迟到15分钟内扣50元，超过30分钟按半天事假。"),
    ("离职制度", "离职：员工需提前30天书面通知，交接完成后发放工资与补偿。"),
    ("培训制度", "培训：新员工入职培训3天，每年至少2次专业培训，培训费用公司承担。"),
    ("差旅制度", "差旅：国内出差每天补贴200元，住宿上限500元/晚，需保存发票。"),
]
names = [d[0] for d in docs]
texts = [d[1] for d in docs]

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
        print(f"[warn] 真实 embedding 不可用，使用本地 fallback。")
        return _local_embed(texts)

def retrieve(q, k=3, threshold=0.15):
    qv = embed([q])
    D, I = index.search(qv, k)
    res = []
    for score, idx in zip(D[0], I[0]):
        if score >= threshold:
            res.append((float(score), idx, names[idx], texts[idx]))
    return res

def answer(q):
    hits = retrieve(q)
    if not hits:
        return "未在知识库中找到相关信息，无法回答。", []
    ctx = "\n".join(f"[{h[2]}] {h[3]}" for h in hits)
    prompt = (f"你是HR助手。基于以下资料回答问题，资料不足请说不知道。"
              f"回答末尾列出引用的制度名。\n\n资料：\n{ctx}\n\n问题：{q}")
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content, hits

# build index
V = embed(texts)
index = faiss.IndexFlatIP(V.shape[1])
index.add(V)

# demo questions
questions = [
    "年假怎么算？",
    "报销超过5000怎么办？",
    "出差补贴多少？",
    "公司有股票期权吗？",  # 不在库中，应降级
]

for q in questions:
    print("=" * 60)
    print("Q:", q)
    ans, hits = answer(q)
    print("检索:", [(round(h[0], 3), h[2]) for h in hits])
    print("A:", ans)
    print()
if __name__ == "__main__":
    pass
