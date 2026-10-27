"""
W5 第 3 课：记忆检索与注入 —— 什么时候调哪条记忆

三种注入策略：
  短期记忆 → 取最近 N 条（滑动窗口）
  程序记忆 → 每次注入系统提示词（始终生效）
  长期记忆 → 检索相关部分注入（不相关的不塞）
"""
import os
import json
import time
import sqlite3
from collections import Counter

DB_PATH = os.path.expanduser("~/hermes/ai-learning/week5/memory_demo.db")


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ══════════════════════════════════════════════════════════
# ① 短期记忆：取最近 N 条
# ══════════════════════════════════════════════════════════
def inject_short_term(history: list, last_n: int = 4) -> list:
    """越近越相关 → 只取最近 N 条"""
    return history[-last_n:]


# ══════════════════════════════════════════════════════════
# ② 程序记忆：全部注入系统提示词（量小，必须始终生效）
# ══════════════════════════════════════════════════════════
def inject_procedural(procedural: dict) -> str:
    if not procedural:
        return ""
    lines = [f"- {k}：{v}" for k, v in procedural.items()]
    return "【用户偏好（始终生效）】\n" + "\n".join(lines)


# ══════════════════════════════════════════════════════════
# ③ 长期记忆：检索相关部分
# ══════════════════════════════════════════════════════════
def retrieve(semantic: dict, episodic: list, query: str, top_k: int = 3) -> dict:
    """检索：关键词命中 + 时间衰减，只返回最相关的"""
    query_terms = set(query)          # 字符级匹配（中文友好）

    scored_sem = []
    for k, v in semantic.items():
        text = f"{k} {v}"
        score = sum(1 for t in query_terms if t in text)   # 命中越多分越高
        if score > 0:
            scored_sem.append((score, f"{k}: {v}"))

    scored_epi = []
    for e in episodic:
        score = sum(1 for t in query_terms if t in e["event"])
        if score > 0:
            # 时间衰减：越近权重越高（这里用序号模拟）
            score += 0.1
            scored_epi.append((score, f"[{e['ts']}] {e['event']}"))

    top_sem = [x[1] for x in sorted(scored_sem, reverse=True)[:top_k]]
    top_epi = [x[1] for x in sorted(scored_epi, reverse=True)[:top_k]]
    return {"相关事实": top_sem, "相关事件": top_epi}


# ══════════════════════════════════════════════════════════
# 演示：对比"全量注入" vs "检索注入"
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("准备：往数据库灌入大量记忆（模拟长期积累）")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("DELETE FROM semantic"); c.execute("DELETE FROM episodic"); c.execute("DELETE FROM procedural")
facts = {"用户姓名": "阿华", "用户职业": "后端开发者", "项目技术栈": "Python + FastAPI",
         "用户城市": "北京", "用户爱好": "篮球", "用户饮食": "不吃辣", "团队规模": "15人",
         "公司年假": "15天", "用户星座": "天蝎座", "用户手机": "iPhone"}
for k, v in facts.items():
    c.execute("INSERT OR REPLACE INTO semantic VALUES(?,?,?)", (k, json.dumps(v, ensure_ascii=False), now()))
events = ["用户询问年假制度", "用户咨询记忆系统设计", "用户讨论RAG检索优化",
          "用户询问加班费计算", "用户咨询Python性能优化", "用户问过旅游推荐",
          "用户讨论数据库选型", "用户咨询简历优化"]
for e in events:
    c.execute("INSERT INTO episodic(ts,event) VALUES(?,?)", (now(), e))
c.execute("INSERT OR REPLACE INTO procedural VALUES(?,?,?)", ("偏好语言", json.dumps("Python"), now()))
c.execute("INSERT OR REPLACE INTO procedural VALUES(?,?,?)", ("讨厌的语言", json.dumps("Java"), now()))
conn.commit()

semantic = {r[0]: json.loads(r[1]) for r in c.execute("SELECT key,value FROM semantic")}
episodic = [{"ts": r[0], "event": r[1]} for r in c.execute("SELECT ts,event FROM episodic")]
procedural = {r[0]: json.loads(r[1]) for r in c.execute("SELECT key,value FROM procedural")}
print(f"  语义记忆 {len(semantic)} 条，情节记忆 {len(episodic)} 条\n")

# 模拟一段很长的历史对话
history = [{"role": "user", "content": f"这是第{i}轮对话内容，讨论一些无关话题……"} for i in range(1, 21)]
history += [{"role": "user", "content": "对了，公司年假到底有多少天？"},
            {"role": "assistant", "content": "正式员工年假15天。"}]

QUERY = "年假还有多少天"

print("=" * 60)
print("方案A：全量注入（把所有记忆都塞进提示词）")
print("=" * 60)
total_chars = len(json.dumps(semantic, ensure_ascii=False)) + len(json.dumps(episodic, ensure_ascii=False))
print(f"  注入 {len(semantic)} 条事实 + {len(episodic)} 条事件 = {total_chars} 字符")
print("  → 问题：90% 的记忆和'年假'无关，白白占上下文 ❌\n")

print("=" * 60)
print("方案B：检索注入（只拿相关的）")
print("=" * 60)
retrieved = retrieve(semantic, episodic, QUERY, top_k=3)
print(f"  查询: 「{QUERY}」\n")
print(f"  ② 程序记忆（系统提示词）:\n    {inject_procedural(procedural).replace(chr(10), chr(10)+'    ')}")
print(f"\n  ③ 长期记忆（检索命中）:")
print(f"    相关事实: {retrieved['相关事实']}")
print(f"    相关事件: {retrieved['相关事件']}")
print(f"\n  ① 短期记忆（最近4条）:")
for m in inject_short_term(history, last_n=4):
    print(f"    [{m['role']}] {m['content']}")

kept_chars = len(json.dumps(retrieved, ensure_ascii=False))
print(f"\n  注入约 {kept_chars} 字符 → 节省 {100 - kept_chars*100//max(total_chars,1)}% 上下文 ✅")

print("\n" + "=" * 60)
print("关键：不是'记住一切'，而是'需要时能找到相关的'。")
print("=" * 60)
