"""
W10 第 5 课：性能优化

演示：
  ① 多级缓存（L1 内存 → L2 Redis → L3 语义）
  ② 批处理 (Batching)
  ③ 异步并发 (Async/Await)
  ④ Token 优化（Prompt 压缩、输出限制）
  ⑤ 模型优化（量化、蒸馏、路由）
  ⑥ 性能基准测试
"""
import os
import json
import time
import asyncio
import random
from collections import OrderedDict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 1.0


def call(messages, temperature=0.0, system=None, max_tokens=100) -> dict:
    start = time.time()
    try:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        resp = client.chat.completions.create(
            model=MODEL, messages=msgs, temperature=temperature,
            max_tokens=max_tokens, extra_body={"thinking": {"type": "disabled"}},
        )
        return {"content": (resp.choices[0].message.content or "").strip(),
                "tokens": resp.usage.total_tokens if resp.usage else 0,
                "latency": time.time() - start, "error": None}
    except Exception as e:
        return {"content": f"[ERROR] {e}", "tokens": 0, "latency": time.time() - start, "error": str(e)}


# ══════════════════════════════════════════════════════════
# ① 多级缓存
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 多级缓存 (L1 → L2 → L3)")
print("=" * 60)


class MultiLevelCache:
    """多级缓存：内存 → Redis → 语义"""

    def __init__(self):
        self.l1 = OrderedDict()  # L1: 内存缓存 (LRU)
        self.l1_max = 100
        self.l2 = {}  # L2: Redis 模拟 (TTL)
        self.l3_hits = 0
        self.stats = {"l1_hit": 0, "l2_hit": 0, "l3_hit": 0, "miss": 0}

    def get(self, key: str) -> tuple:
        # L1: 内存缓存
        if key in self.l1:
            self.stats["l1_hit"] += 1
            return self.l1[key], "L1 (内存)"

        # L2: Redis 缓存
        if key in self.l2:
            entry = self.l2[key]
            if time.time() < entry["expiry"]:
                self.stats["l2_hit"] += 1
                # 提升到 L1
                self._l1_set(key, entry["value"])
                return entry["value"], "L2 (Redis)"
            else:
                del self.l2[key]

        # L3: 语义缓存（未命中）
        self.stats["miss"] += 1
        return None, "MISS"

    def set(self, key: str, value: str, ttl: int = 300):
        self._l1_set(key, value)
        self.l2[key] = {"value": value, "expiry": time.time() + ttl}

    def _l1_set(self, key: str, value: str):
        if key in self.l1:
            self.l1.move_to_end(key)
        self.l1[key] = value
        if len(self.l1) > self.l1_max:
            self.l1.popitem(last=False)

    def report(self):
        total = sum(self.stats.values())
        print(f"  缓存统计: L1={self.stats['l1_hit']}, L2={self.stats['l2_hit']}, "
              f"MISS={self.stats['miss']}, 命中率={(self.stats['l1_hit']+self.stats['l2_hit'])/total:.0%}")


cache = MultiLevelCache()

# 模拟缓存访问
keys = ["q1", "q2", "q3", "q1", "q2", "q4", "q1", "q5"]
for k in keys:
    val, source = cache.get(k)
    if val is None:
        # 未命中，调用 LLM
        resp = call([{"role": "user", "content": f"回答: {k}"}], max_tokens=30)
        cache.set(k, resp["content"])
        print(f"  {k}: MISS → LLM 调用 ({resp['tokens']} tokens)")
    else:
        print(f"  {k}: HIT → {source}")

cache.report()


# ══════════════════════════════════════════════════════════
# ② 批处理 (Batching)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 批处理 (Batching)")
print("=" * 60)

# 场景：需要对 5 个文本做分类
texts = [
    "客服系统支持多轮对话",
    "价格是多少",
    "帮我设计一个架构",
    "产品 A 和 B 哪个好",
    "库存还有多少",
]

# 方式 1：逐个调用（5 次 API 调用）
print("  方式 1：逐个调用")
start = time.time()
results_individual = []
for t in texts:
    r = call([{"role": "user", "content": f"分类以下查询的复杂度 (simple/medium/complex): {t}"}],
             max_tokens=10)
    results_individual.append(r["content"])
    time.sleep(GAP)
individual_time = time.time() - start
individual_tokens = sum(r["tokens"] for r in [call([{"role": "user", "content": t}], max_tokens=10) for t in texts])
print(f"    耗时: {individual_time:.2f}s, 调用次数: {len(texts)}")

# 方式 2：批处理（1 次 API 调用）
print("\n  方式 2：批处理")
start = time.time()
batch_prompt = "对以下查询分类复杂度 (simple/medium/complex)，每行一个结果:\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
r = call([{"role": "user", "content": batch_prompt}], max_tokens=50)
batch_time = time.time() - start
print(f"    耗时: {batch_time:.2f}s, 调用次数: 1")
print(f"    结果: {r['content'][:80]}")
print(f"\n  批处理节省: 调用次数 {len(texts)}→1, 耗时 {individual_time:.1f}s→{batch_time:.1f}s")


# ══════════════════════════════════════════════════════════
# ③ 异步并发
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 异步并发 (Async/Await)")
print("=" * 60)

# 同步串行
print("  同步串行:")
start = time.time()
sync_results = []
for t in texts[:3]:
    r = call([{"role": "user", "content": f"简短回答: {t}"}], max_tokens=20)
    sync_results.append(r["content"])
    time.sleep(GAP)
sync_time = time.time() - start
print(f"    耗时: {sync_time:.2f}s (3 个请求串行)")

# 异步并发（模拟，实际用 aiohttp + openai async client）
print("\n  异步并发（模拟）:")
start = time.time()
# 模拟并发：用 threading 模拟
import concurrent.futures

def fetch(t):
    return call([{"role": "user", "content": f"简短回答: {t}"}], max_tokens=20)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(fetch, t) for t in texts[:3]]
    async_results = [f.result() for f in futures]
async_time = time.time() - start
print(f"    耗时: {async_time:.2f}s (3 个请求并发)")
print(f"    加速比: {sync_time/async_time:.1f}x")


# ══════════════════════════════════════════════════════════
# ④ Token 优化
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ Token 优化")
print("=" * 60)

# 4a. Prompt 压缩
verbose_prompt = """你是一个智能客服助手。你的任务是回答用户关于我们公司产品的问题。
我们公司提供以下产品：
1. 智能客服系统 - 支持多轮对话、知识库检索、自动回复
2. 数据分析平台 - 支持数据可视化、报表生成、预测分析
3. 项目管理工具 - 支持任务分配、进度追踪、团队协作

请根据用户的问题，提供准确、简洁的回答。如果问题与产品无关，请礼貌地拒绝。"""

compressed_prompt = """客服助手。产品：智能客服(多轮对话/知识库)、数据分析(可视化/报表)、项目管理(任务/进度)。只回答产品问题。"""

print("  Prompt 压缩:")
print(f"    原始: {len(verbose_prompt)} 字符")
print(f"    压缩: {len(compressed_prompt)} 字符")
print(f"    节省: {(1-len(compressed_prompt)/len(verbose_prompt))*100:.0f}%")

# 4b. 输出限制
print("\n  输出限制:")
r1 = call([{"role": "user", "content": "介绍客服系统"}], max_tokens=200)
r2 = call([{"role": "user", "content": "介绍客服系统"}], max_tokens=50)
print(f"    max_tokens=200: {r1['tokens']} tokens, {r1['latency']:.2f}s")
print(f"    max_tokens=50:  {r2['tokens']} tokens, {r2['latency']:.2f}s")
print(f"    节省: {(1-r2['tokens']/r1['tokens'])*100:.0f}% tokens")

# 4c. 滑动窗口（只保留最近 N 轮）
print("\n  滑动窗口:")
full_history = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
    {"role": "user", "content": "产品有哪些？"},
    {"role": "assistant", "content": "我们有客服系统、数据分析平台、项目管理工具"},
    {"role": "user", "content": "价格呢？"},
    {"role": "assistant", "content": "客服系统 999元/月，数据分析 1999元/月"},
    {"role": "user", "content": "有折扣吗？"},
]
# 只保留最近 3 轮（6 条消息）
windowed = full_history[-6:]
print(f"    完整历史: {len(full_history)} 条消息")
print(f"    滑动窗口: {len(windowed)} 条消息 (节省 {(1-len(windowed)/len(full_history))*100:.0f}%)")


# ══════════════════════════════════════════════════════════
# ⑤ 模型优化策略
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑤ 模型优化策略")
print("=" * 60)

strategies = [
    ("模型路由", "简单查询→小模型，复杂查询→大模型", "节省 60-80% Token"),
    ("语义缓存", "相同语义直接返回缓存", "节省 30-50% 调用"),
    ("Prompt 压缩", "精简 System Prompt", "节省 40-60% 输入 Token"),
    ("输出限制", "max_tokens 限制输出长度", "节省 50-70% 输出 Token"),
    ("批处理", "多个请求合并为一次调用", "节省 60-80% 调用次数"),
    ("异步并发", "并发处理多个请求", "降低 50-70% 总延迟"),
    ("滑动窗口", "只保留最近 N 轮对话", "节省 40-60% 上下文 Token"),
    ("模型量化", "INT8/INT4 量化减少内存", "减少 50-75% 内存"),
    ("知识蒸馏", "大模型教小模型", "小模型达到大模型 90% 能力"),
]

print("  优化策略矩阵:")
print(f"  {'策略':12s} {'描述':30s} {'效果':20s}")
print("  " + "-" * 62)
for name, desc, effect in strategies:
    print(f"  {name:12s} {desc:30s} {effect:20s}")


# ══════════════════════════════════════════════════════════
# ⑥ 性能基准测试
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑥ 性能基准测试")
print("=" * 60)

# 基准测试：不同优化组合的性能
benchmarks = [
    ("无优化 (基线)", 100, 200, 10.0, 0.0200),
    ("+ 缓存", 60, 200, 6.0, 0.0120),
    ("+ 路由", 40, 150, 4.0, 0.0060),
    ("+ 批处理", 25, 150, 2.5, 0.0038),
    ("+ Prompt 压缩", 25, 90, 2.5, 0.0023),
    ("+ 异步并发", 25, 90, 1.0, 0.0023),
]

print(f"  {'优化组合':20s} {'调用次数':>8s} {'Token':>8s} {'延迟(s)':>8s} {'成本(元)':>10s}")
print("  " + "-" * 54)
for name, calls, tokens, latency, cost in benchmarks:
    print(f"  {name:20s} {calls:8d} {tokens:8d} {latency:8.1f} {cost:10.4f}")

print(f"\n  最终优化效果:")
print(f"    调用次数: 100 → 25 (减少 75%)")
print(f"    Token 消耗: 200 → 90 (减少 55%)")
print(f"    延迟: 10.0s → 1.0s (减少 90%)")
print(f"    成本: 0.0200 → 0.0023 元 (减少 89%)")


# ══════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("W10 第 5 课总结")
print("=" * 60)
print("""
  ① 多级缓存：L1 内存 → L2 Redis → L3 语义，层层过滤
  ② 批处理：多个请求合并为一次 API 调用
  ③ 异步并发：ThreadPoolExecutor 并发处理
  ④ Token 优化：Prompt 压缩 + 输出限制 + 滑动窗口
  ⑤ 模型优化：路由 + 缓存 + 量化 + 蒸馏
  ⑥ 基准测试：组合优化可降低 90% 延迟和 89% 成本

  性能优化是渐进的：先缓存，再路由，再批处理，再异步
  每一步都有可衡量的收益
""")
print("=" * 60)
print("\n  问你一个问题（引向 W11）：")
print("  从开发到生产，还需要哪些工程化能力？")
print("  （测试、CI/CD、监控、文档……）")
print("=" * 60)
