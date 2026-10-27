"""
W5 第 2 课：结构化记忆存储 —— 让记忆跨会话"落盘"

用 SQLite（文件型数据库）：不用装服务，但能 SQL 查询。
用法：
    python3 02_memory_storage.py save    # 会话1：写入记忆
    python3 02_memory_storage.py load    # 会话2：读取记忆（模拟重启）
"""
import sys
import os
import json
import time
import sqlite3

DB_PATH = os.path.expanduser("~/hermes/ai-learning/week5/memory_demo.db")


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class MemoryStore:
    """记忆持久化存储（SQLite）"""

    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path)
        self._init_schema()

    def _init_schema(self):
        c = self.conn.cursor()
        # 语义记忆：事实
        c.execute("""CREATE TABLE IF NOT EXISTS semantic(
                     key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)""")
        # 情节记忆：历史事件（带时间线）
        c.execute("""CREATE TABLE IF NOT EXISTS episodic(
                     id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, event TEXT)""")
        # 程序记忆：偏好
        c.execute("""CREATE TABLE IF NOT EXISTS procedural(
                     key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)""")
        self.conn.commit()

    # ── 语义记忆 ──
    def save_fact(self, key, value):
        self.conn.execute("INSERT OR REPLACE INTO semantic VALUES(?,?,?)",
                          (key, json.dumps(value, ensure_ascii=False), now()))
        self.conn.commit()

    def load_semantic(self):
        return {r[0]: json.loads(r[1]) for r in self.conn.execute("SELECT key,value FROM semantic")}

    # ── 情节记忆 ──
    def add_event(self, event):
        self.conn.execute("INSERT INTO episodic(ts,event) VALUES(?,?)", (now(), event))
        self.conn.commit()

    def load_episodic(self, limit=50):
        return [{"ts": r[0], "event": r[1]}
                for r in self.conn.execute("SELECT ts,event FROM episodic ORDER BY id DESC LIMIT ?", (limit,))]

    # ── 程序记忆 ──
    def save_preference(self, key, value):
        self.conn.execute("INSERT OR REPLACE INTO procedural VALUES(?,?,?)",
                          (key, json.dumps(value, ensure_ascii=False), now()))
        self.conn.commit()

    def load_procedural(self):
        return {r[0]: json.loads(r[1]) for r in self.conn.execute("SELECT key,value FROM procedural")}

    def dump(self):
        return {"语义记忆": self.load_semantic(),
                "情节记忆": self.load_episodic(),
                "程序记忆": self.load_procedural()}


# ══════════════════════════════════════════════════════════
# 演示：两个独立进程（模拟两次会话）
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "load"
    store = MemoryStore()

    if mode == "save":
        print("【会话 1】写入记忆...\n")
        store.save_fact("用户姓名", "阿华")
        store.save_fact("用户职业", "后端开发者")
        store.save_fact("项目技术栈", "Python + FastAPI")
        store.save_preference("偏好语言", "Python")
        store.save_preference("讨厌的语言", "Java")
        store.add_event("用户询问年假制度")
        store.add_event("用户咨询记忆系统设计")
        print("已写入 3 条事实、2 条偏好、2 条事件")
        print(f"落盘位置: {DB_PATH}")

    else:
        print("【会话 2】（模拟重启后）读取记忆...\n")
        data = store.dump()
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("\n✅ 关键：进程重启了，但记忆还在 —— 这就是'跨会话'。")
