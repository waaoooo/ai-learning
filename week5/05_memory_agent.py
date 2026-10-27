"""
W5 第 5 课：综合 —— 带记忆管理功能的完整 Agent

整合前四课：
  记忆分类（T1）→ 结构化存储（T2）→ 检索注入（T3）→ 更新遗忘（T4）
  + 记忆管理（T5）：让 LLM 自己判断"这条值得记吗"

流程：
  用户消息 → 检索相关记忆 → 组装提示词 → 调 LLM → 记忆管理（提取该记的）
"""
import os
import json
import time
import sqlite3
from datetime import datetime, timedelta

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

DB_PATH = os.path.expanduser("~/hermes/ai-learning/week5/agent_memory.db")
DAY = 86400


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ts(days_ago=0):
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


# ══════════════════════════════════════════════════════════
# 记忆层（T1 分类 + T2 存储 + T3 检索 + T4 更新遗忘）
# ══════════════════════════════════════════════════════════
class Memory:
    """四类记忆 + 持久化 + 检索 + 更新遗忘"""

    def __init__(self, path=DB_PATH):
        # isolation_level=None → 自动提交模式：每条语句立即提交，不留写锁
        self.conn = sqlite3.connect(path, timeout=10, isolation_level=None)
        self.conn.execute("PRAGMA busy_timeout=10000")   # 锁冲突时等待而非报错
        self.conn.execute("PRAGMA journal_mode=WAL")     # 读写并发
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS semantic(
                     key TEXT PRIMARY KEY, value TEXT, importance REAL, updated_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS procedural(
                     key TEXT PRIMARY KEY, value TEXT, importance REAL, updated_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS episodic(
                     id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, event TEXT)""")
        self.conn.commit()

    # ── 写入（T4 更新：覆盖，不堆积）──
    def remember(self, table, key, value, importance=0.5):
        tbl = "procedural" if table == "preference" else "semantic"
        self.conn.execute(f"INSERT OR REPLACE INTO {tbl} VALUES(?,?,?,?)",
                          (key, json.dumps(value, ensure_ascii=False), importance, now()))
        self.conn.commit()

    def add_event(self, event):
        self.conn.execute("INSERT INTO episodic(ts,event) VALUES(?,?)", (now(), event))

    # ── 读取（T3 检索：只拿相关的）──
    def retrieve(self, query, top_k=3):
        terms = set(query)
        sem = {r[0]: (json.loads(r[1]), r[2]) for r in
               self.conn.execute("SELECT key,value,importance FROM semantic")}
        proc = {r[0]: (json.loads(r[1]), r[2]) for r in
                self.conn.execute("SELECT key,value,importance FROM procedural")}

        scored = []
        for k, (v, imp) in {**sem, **proc}.items():
            hit = sum(1 for t in terms if t in f"{k} {v}")
            if hit > 0:
                scored.append((hit, f"{k}: {v}"))
        scored.sort(reverse=True)
        return {
            "程序记忆（始终注入）": [f"{k}: {v}" for k, (v, _) in proc.items()],
            "相关事实": [x[1] for x in scored[:top_k]],
        }

    def forget(self, threshold=0.05, protect=0.9):
        removed = []
        for tbl in ("semantic", "procedural"):
            for key, imp, upd in self.conn.execute(
                    f"SELECT key,importance,updated_at FROM {tbl}").fetchall():
                if imp >= protect:
                    continue
                age = (datetime.now() - datetime.strptime(upd, "%Y-%m-%d %H:%M:%S")).days
                score = imp / (1 + age)
                if score < threshold:
                    self.conn.execute(f"DELETE FROM {tbl} WHERE key=?", (key,))
                    removed.append(key)
        self.conn.commit()
        return removed

    def dump(self):
        sem = {r[0]: json.loads(r[1]) for r in self.conn.execute("SELECT key,value FROM semantic")}
        proc = {r[0]: json.loads(r[1]) for r in self.conn.execute("SELECT key,value FROM procedural")}
        epi = [r[0] for r in self.conn.execute("SELECT event FROM episodic")]
        return {"语义记忆": sem, "程序记忆": proc, "情节记忆": epi}


# ══════════════════════════════════════════════════════════
# LLM 层
# ══════════════════════════════════════════════════════════
class LLM:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                             base_url="https://open.bigmodel.cn/api/paas/v4/")

    def call(self, messages, json_mode=False):
        for attempt in range(4):
            try:
                kw = dict(model="glm-4.7-flash", messages=messages, temperature=0,
                          max_tokens=500, extra_body={"thinking": {"type": "disabled"}})
                if json_mode:
                    kw["response_format"] = {"type": "json_object"}
                return self.client.chat.completions.create(**kw).choices[0].message.content
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)


# ══════════════════════════════════════════════════════════
# T5 核心：记忆管理 —— 让 LLM 自己判断"该记什么"
# ══════════════════════════════════════════════════════════
MANAGE_PROMPT = """你是记忆管理模块。从对话中提取"值得长期记住"的信息。

只提取这三类：
1. 用户事实（姓名、职业、城市等）
2. 用户偏好（喜欢/讨厌什么）
3. 重要决定（用户明确做出的选择）

规则：
- 不提取闲聊、问候、一次性问题
- importance 范围 0.1~1.0（姓名/关键偏好 0.8-1.0，一般信息 0.3-0.6）
- 没有值得记的就返回空列表

只输出 JSON：
{"facts": [{"key": "...", "value": "...", "importance": 0.8}],
 "preferences": [{"key": "...", "value": "...", "importance": 0.8}]}"""


class MemoryAgent:
    def __init__(self):
        self.llm = LLM()
        self.memory = Memory()
        self.history = []

    def chat(self, user_msg):
        # ① 检索相关记忆
        relevant = self.memory.retrieve(user_msg)

        # ② 组装提示词（程序记忆进系统提示词，相关事实注入）
        sys_parts = ["你是助手，只能根据已知信息回答，不要编造。"]
        if relevant["程序记忆（始终注入）"]:
            sys_parts.append("【用户偏好（始终生效）】\n" +
                             "\n".join(f"- {x}" for x in relevant["程序记忆（始终注入）"]))
        if relevant["相关事实"]:
            sys_parts.append("【相关记忆】\n" +
                             "\n".join(f"- {x}" for x in relevant["相关事实"]))

        messages = [{"role": "system", "content": "\n".join(sys_parts)}] + self.history[-4:] + \
                   [{"role": "user", "content": user_msg}]

        # ③ 调 LLM
        answer = self.llm.call(messages)
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": answer})

        # ④ 记忆管理：LLM 判断该记什么
        self._manage(user_msg, answer)
        self.memory.add_event(f"用户：{user_msg[:20]}")
        return answer

    def _manage(self, user_msg, answer):
        raw = self.llm.call([{"role": "system", "content": MANAGE_PROMPT},
                             {"role": "user", "content": f"用户：{user_msg}\n助手：{answer}"}],
                            json_mode=True)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        for f in data.get("facts", []):
            self.memory.remember("fact", f["key"], f["value"], f.get("importance", 0.5))
            print(f"      🧠 记住事实: {f['key']} = {f['value']}")
        for p in data.get("preferences", []):
            self.memory.remember("preference", p["key"], p["value"], p.get("importance", 0.8))
            print(f"      🧠 记住偏好: {p['key']} = {p['value']}")


# ══════════════════════════════════════════════════════════
# 演示：跨会话记住用户
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print("=" * 60)
    print("带记忆管理的 Agent —— 演示")
    print("=" * 60)
    agent = MemoryAgent()

    print("\n【会话 1】用户自我介绍\n")
    for q in ["我叫阿华，是个后端开发者，在北京工作。",
              "我特别讨厌 Java，平时写 Python。"]:
        print(f"  用户: {q}")
        ans = agent.chat(q)
        print(f"  助手: {ans}\n")

    print("=" * 60)
    print("【会话 2】新开会话，看它记不记得")
    print("=" * 60)
    agent.memory.conn.close()      # 关掉会话1的连接（模拟进程结束）
    agent2 = MemoryAgent()        # 全新实例，历史为空
    print("\n  当前记忆库:")
    print(json.dumps(agent2.memory.dump(), ensure_ascii=False, indent=2))

    print("\n  用户: 帮我推荐一门编程语言学习")
    ans = agent2.chat("帮我推荐一门编程语言学习")
    print(f"  助手: {ans}")
    print("\n  ✅ 注意：它记得你偏好 Python、讨厌 Java —— 历史是空的，但记忆还在。")

    print("\n" + "=" * 60)
    print("记忆管理 = LLM 自己判断'这条值得记吗'，不靠用户说'请记住'。")
    print("=" * 60)
