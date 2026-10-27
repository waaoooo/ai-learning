"""
W5 第 4 课：记忆更新与遗忘

更新：新信息覆盖旧信息（冲突检测 + 覆盖）
遗忘：时间衰减 + 重要性加权，低于阈值就删
"""
import os
import json
import time
import sqlite3
from datetime import datetime, timedelta

DB_PATH = os.path.expanduser("~/hermes/ai-learning/week5/memory_update.db")
DAY = 86400


def ts(days_ago=0):
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


class MemoryManager:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path)
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS memory(
                     key TEXT PRIMARY KEY, value TEXT, importance REAL,
                     created_at TEXT, updated_at TEXT, access_count INTEGER DEFAULT 0)""")
        self.conn.commit()

    def write(self, key, value, importance=0.5, days_ago=0):
        c = self.conn.cursor()
        row = c.execute("SELECT value FROM memory WHERE key=?", (key,)).fetchone()
        if row:
            # ── 更新：覆盖旧值（记录这是第几次更新）──
            c.execute("""UPDATE memory SET value=?, importance=?, updated_at=?, access_count=access_count+1
                         WHERE key=?""", (json.dumps(value, ensure_ascii=False), importance, ts(), key))
            action = f"更新（原值: {json.loads(row[0])}）"
        else:
            c.execute("""INSERT INTO memory VALUES(?,?,?,?,?,0)""",
                      (key, json.dumps(value, ensure_ascii=False), importance, ts(days_ago), ts(days_ago)))
            action = "新增"
        self.conn.commit()
        return action

    def read(self, key):
        c = self.conn.cursor()
        c.execute("UPDATE memory SET access_count=access_count+1 WHERE key=?", (key,))
        self.conn.commit()
        row = c.execute("SELECT value FROM memory WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    # ══════════════════════════════════════════════════════
    # 遗忘策略：时间衰减 × 重要性 × 使用频率
    # ══════════════════════════════════════════════════════
    def forget(self, threshold=0.05, protect_importance=0.9):
        """score 越低越该被遗忘；低于 threshold 就删。
        高重要性记忆（>= protect_importance）有保护期，不参与遗忘 ——
        否则'用户姓名'这类关键事实会被时间衰减误删。"""
        c = self.conn.cursor()
        removed = []
        for key, imp, updated, cnt in c.execute(
                "SELECT key,importance,updated_at,access_count FROM memory").fetchall():
            if imp >= protect_importance:          # ★ 保护期：关键事实不遗忘
                continue
            age_days = (datetime.now() - datetime.strptime(updated, "%Y-%m-%d %H:%M:%S")).days
            # 分数 = 重要性 / (1 + 天数) × (1 + 使用次数)
            score = (imp / (1 + age_days)) * (1 + cnt)
            if score < threshold:
                c.execute("DELETE FROM memory WHERE key=?", (key,))
                removed.append((key, round(score, 4)))
        self.conn.commit()
        return removed

    def dump(self):
        c = self.conn.cursor()
        rows = c.execute("""SELECT key,value,importance,updated_at,access_count
                            FROM memory ORDER BY importance DESC""").fetchall()
        return [{"key": r[0], "value": json.loads(r[1]), "重要性": r[2],
                 "更新时间": r[3], "使用次数": r[4]} for r in rows]


# ══════════════════════════════════════════════════════════
# 演示
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    m = MemoryManager()

    print("=" * 60)
    print("① 更新：用户改主意了 → 覆盖旧记忆")
    print("=" * 60)
    m.write("偏好语言", "Python", importance=0.9)
    print(f"  第一次: 偏好语言 = Python")
    action = m.write("偏好语言", "Go", importance=0.9)
    print(f"  用户说「我改用 Go 了」→ {action}")
    print(f"  现在: 偏好语言 = {m.read('偏好语言')}")
    print("  ✅ 旧值被覆盖，没有堆积成两条互相矛盾的记忆\n")

    print("=" * 60)
    print("② 遗忘：太久、不重要的记忆该清理掉")
    print("=" * 60)
    m.write("临时事项", "记得买牛奶", importance=0.2, days_ago=40)      # 很久 + 不重要
    m.write("用户星座", "天蝎座", importance=0.1, days_ago=30)          # 很久 + 不重要
    m.write("用户姓名", "阿华", importance=1.0, days_ago=60)            # 很久但很重要
    m.write("项目技术栈", "Python + FastAPI", importance=0.8, days_ago=5)
    m.write("用户城市", "北京", importance=0.7, days_ago=10)

    print("\n  遗忘前:")
    for r in m.dump():
        print(f"    {r['key']}: {r['value']}  (重要性{r['重要性']}, 使用{r['使用次数']}次)")

    removed = m.forget(threshold=0.05)
    print(f"\n  已遗忘 {len(removed)} 条:")
    for k, s in removed:
        print(f"    ✗ {k}  (分数 {s})")

    print("\n  遗忘后:")
    for r in m.dump():
        print(f"    ✓ {r['key']}: {r['value']}  (重要性{r['重要性']}, 使用{r['使用次数']}次)")

    print("\n" + "=" * 60)
    print("关键：更新 = 覆盖（不堆积矛盾）；遗忘 = 按分数衰减（不无限膨胀）。")
    print("      分数 = 重要性 / (1+天数) × (1+使用次数)")
    print("=" * 60)
