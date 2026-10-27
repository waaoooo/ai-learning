"""
W5 第 1 课：记忆分类 —— 先搞清"记什么"

四类记忆：
  短期记忆  —— 当前对话上下文（W1-W4 只有这个）
  语义记忆  —— 事实、知识
  情节记忆  —— 历史事件
  程序记忆  —— 偏好、习惯
"""
import json
import time


class Memory:
    """四类记忆的雏形"""

    def __init__(self):
        self.short_term = []     # 短期：当前对话消息
        self.semantic = {}       # 语义：事实（key-value）
        self.episodic = []       # 情节：历史事件（时间线）
        self.procedural = {}     # 程序：偏好/习惯

    # ── 短期记忆 ──
    def add_message(self, role, content):
        self.short_term.append({"role": role, "content": content})

    def get_short_term(self, last_n=None):
        return self.short_term[-last_n:] if last_n else self.short_term

    # ── 语义记忆：事实 ──
    def remember_fact(self, key, value):
        self.semantic[key] = value

    def recall_fact(self, key):
        return self.semantic.get(key)

    # ── 情节记忆：历史事件 ──
    def remember_event(self, event):
        self.episodic.append({"ts": time.strftime("%Y-%m-%d %H:%M"), "event": event})

    def recall_events(self, keyword=None):
        if not keyword:
            return self.episodic
        return [e for e in self.episodic if keyword in e["event"]]

    # ── 程序记忆：偏好/习惯 ──
    def remember_preference(self, key, value):
        self.procedural[key] = value

    def recall_preference(self, key):
        return self.procedural.get(key)

    def dump(self):
        return {
            "短期记忆（当前对话）": len(self.short_term),
            "语义记忆（事实）": self.semantic,
            "情节记忆（历史事件）": self.episodic,
            "程序记忆（偏好）": self.procedural,
        }


# ══════════════════════════════════════════════════════════
# 演示：四类记忆分别怎么用
# ══════════════════════════════════════════════════════════
mem = Memory()

print("=" * 60)
print("① 短期记忆：当前对话")
print("=" * 60)
mem.add_message("user", "我叫阿华，买了一件衣服")
mem.add_message("assistant", "收到，请问尺码合适吗？")
for m in mem.get_short_term():
    print(f"  [{m['role']}] {m['content']}")

print("\n" + "=" * 60)
print("② 语义记忆：事实（跨会话也能记住）")
print("=" * 60)
mem.remember_fact("用户职业", "后端开发者")
mem.remember_fact("用户姓名", "阿华")
print(f"  用户职业: {mem.recall_fact('用户职业')}")
print(f"  用户姓名: {mem.recall_fact('用户姓名')}")

print("\n" + "=" * 60)
print("③ 情节记忆：历史事件（时间线）")
print("=" * 60)
mem.remember_event("用户询问年假制度")
mem.remember_event("用户购买了一件衣服")
mem.remember_event("用户问过年假和病假哪个更长")
for e in mem.recall_events():
    print(f"  {e['ts']}  {e['event']}")

print("\n" + "=" * 60)
print("④ 程序记忆：偏好/习惯")
print("=" * 60)
mem.remember_preference("讨厌的语言", "Java")
mem.remember_preference("偏好语言", "Python")
print(f"  讨厌: {mem.recall_preference('讨厌的语言')}")
print(f"  偏好: {mem.recall_preference('偏好语言')}")

print("\n" + "=" * 60)
print("记忆全景")
print("=" * 60)
print(json.dumps(mem.dump(), ensure_ascii=False, indent=2))

print("\n" + "=" * 60)
print("关键：不同类型记忆，读取时机和方式完全不同 ——")
print("      这就是记忆系统的核心设计问题。")
print("=" * 60)
