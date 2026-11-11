# -*- coding: utf-8 -*-
"""L2-06 混合检索：BM25 + 向量 + RRF（医疗实体）"""
import os, hashlib, math, numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../../week1/.env")
client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4")

# 医疗实体库
docs = [
    "高血压 (Hypertension)：血压持续升高，常见症状头痛、头晕，药物：氨氯地平、缬沙坦。",
    "2型糖尿病 (T2DM)：血糖代谢异常，药物：二甲双胍、格列齐特；监测 HbA1c。",
    "冠心病 (CHD)：冠状动脉狭窄，症状胸痛、心悸，药物：阿司匹林、阿托伐他汀。",
    "哮喘 (Asthma)：气道慢性炎症，症状喘息、胸闷，药物：沙丁胺醇、布地奈德。",
    "肺炎 (Pneumonia)：肺部感染，症状发热、咳嗽、胸痛，抗生素治疗。",
    "胃溃疡 (Peptic Ulcer)：胃/十二指肠黏膜缺损，药物：奥美拉唑、阿莫西林。",
    "脑卒中 (Stroke)：脑血管急性损伤，FAST 识别：面部下垂、手臂无力、言语困难。",
    "慢性肾病 (CKD)：肾功能渐进性下降，监测肌酐、eGFR，饮食控制。",
    "类风湿关节炎 (RA)：自身免疫性关节炎症，药物：甲氨蝶呤、TNF抑制剂。",
    "抑郁障碍 (MDD)：心境低落、兴趣减退，SSRI 一线治疗。",
    "甲状腺功能亢进 (Hyperthyroidism)：甲亢，症状心悸、体重下降，药物：甲巯咪唑。",
    "骨质疏松 (Osteoporosis)：骨密度降低，补充钙+维生素D，双膦酸盐。",
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

# ---------- BM25 (char-level) ----------
def tokenize(s):
    return list(s)  # 字符级

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = [tokenize(d) for d in corpus]
        self.k1, self.b = k1, b
        self.N = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / self.N
        self.df = {}
        for d in self.corpus:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {t: math.log((self.N - df + 0.5) / (df + 0.5) + 1) for t, df in self.df.items()}

    def score(self, query):
        q = tokenize(query)
        scores = []
        for d in self.corpus:
            dl = len(d)
            tf_map = {}
            for t in d:
                tf_map[t] = tf_map.get(t, 0) + 1
            s = 0.0
            for t in q:
                if t not in self.idf:
                    continue
                tf = tf_map.get(t, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf[t] * (tf * (self.k1 + 1)) / (denom + 1e-9)
            scores.append(s)
        return np.array(scores)

def rrf(rank_lists, k=60):
    scores = np.zeros(len(docs))
    for rl in rank_lists:
        for rank, idx in enumerate(rl):
            scores[idx] += 1.0 / (k + rank + 1)
    return scores

def topk_indices(scores, k=3):
    return list(np.argsort(scores)[::-1][:k])

def main():
    V = embed(docs)
    bm25 = BM25(docs)

    queries = [
        "高血压用药",
        "T2DM 血糖",
        "胸痛 心悸",
        "FAST 脑卒中",
    ]

    for q in queries:
        print("=" * 70)
        print("Q:", q)
        # vector
        qv = embed([q])[0]
        vscores = V @ qv
        v_top = topk_indices(vscores)
        # bm25
        bscores = bm25.score(q)
        b_top = topk_indices(bscores)
        # hybrid RRF
        hybrid_scores = rrf([v_top, b_top])
        h_top = topk_indices(hybrid_scores)

        def show(label, idxs, sc=None):
            print(f"\n--- {label} top-3 ---")
            for i in idxs:
                extra = f" (score={sc[i]:.3f})" if sc is not None else ""
                print(f"  [{i}] {docs[i][:38]}{extra}")

        show("纯向量", v_top, vscores)
        show("纯BM25", b_top, bscores)
        show("混合RRF", h_top, hybrid_scores)

if __name__ == "__main__":
    main()
