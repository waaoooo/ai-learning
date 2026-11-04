"""
W10 第 1 课：多模型路由与缓存策略

演示：
  ① 意图分类：根据用户输入判断复杂度
  ② 模型路由：简单→小模型，复杂→大模型
  ③ 语义缓存：相同语义直接返回缓存
  ④ 成本对比：路由前后的 Token 消耗
  ⑤ 降级策略：主模型失败 → 备用模型
"""
import os
import re
import json
import time
import hashlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL_SMALL = "glm-4-flash"       # 小模型：便宜快
MODEL_LARGE = "glm-4-flash"        # 大模型：能力强（免费 tier 只有 flash）
GAP = 2.0


def clean_json(text: str) -> dict:
    """清洗 LLM 输出中的 JSON（处理 markdown 包装）"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


def call(messages, model=MODEL_SMALL, temperature=0.0, system=None, max_tokens=200) -> dict:
    """调用 LLM，返回 {content, tokens, latency}"""
    start = time.time()
    last_err = None
    for attempt in range(3):
        try:
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            resp = client.chat.completions.create(
                model=model, messages=msgs, temperature=temperature,
                max_tokens=max_tokens, extra_body={"thinking": {"type": "disabled"}},
            )
            latency = time.time() - start
            content = (resp.choices[0].message.content or "").strip()
            tokens = resp.usage.total_tokens if resp.usage else 0
            return {"content": content, "tokens": tokens, "latency": latency, "model": model, "error": None}
        except Exception as e:
            last_err = e
            if "contentFilter" in str(e) or "1301" in str(e):
                return {"content": "[CONTENT_FILTERED]", "tokens": 0, "latency": time.time()-start,
                        "model": model, "error": str(e)}
            time.sleep(2 ** attempt)
    return {"content": f"[ERROR] {last_err}", "tokens": 0, "latency": time.time()-start,
            "model": model, "error": str(last_err)}


# ══════════════════════════════════════════════════════════
# ① 意图分类：判断查询复杂度
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 意图分类：判断查询复杂度")
print("=" * 60)

INTENT_PROMPT = """判断用户查询的复杂度，输出 JSON。
查询：{query}

复杂度定义：
- simple: 简单事实查询（产品功能、价格、库存）
- medium: 需要推理或比较（产品对比、推荐）
- complex: 需要多步推理或生成（方案设计、代码生成、数据分析）

输出 JSON：{{"complexity": "simple/medium/complex", "reason": "理由"}}"""

test_queries = [
    "客服系统支持哪些功能？",
    "产品 A 和产品 B 哪个更适合中小企业？",
    "帮我设计一个基于 RAG 的客服系统架构，包括技术选型和部署方案",
    "价格是多少？",
    "对比三个产品的优缺点，给出推荐",
]

for q in test_queries:
    r = call([{"role": "user", "content": INTENT_PROMPT.format(query=q)}],
             system="你是查询分类器，只输出 JSON。")
    result = clean_json(r["content"])
    if result:
        icon = {"simple": "🟢", "medium": "🟡", "complex": "🔴"}.get(result["complexity"], "⚪")
        print(f"  {icon} [{result['complexity']}] {q[:40]}")
        print(f"     理由: {result['reason']}")
    else:
        print(f"  ⚠️ 解析失败: {r['content'][:60]}")
    time.sleep(GAP)


# ══════════════════════════════════════════════════════════
# ② 模型路由：根据复杂度选择模型
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 模型路由：根据复杂度选择模型")
print("=" * 60)

ROUTING_TABLE = {
    "simple": {"model": MODEL_SMALL, "max_tokens": 100, "temperature": 0.0},
    "medium": {"model": MODEL_SMALL, "max_tokens": 200, "temperature": 0.3},
    "complex": {"model": MODEL_LARGE, "max_tokens": 500, "temperature": 0.5},
}

def classify_and_route(query: str) -> dict:
    """分类查询并路由到合适的模型"""
    r = call([{"role": "user", "content": INTENT_PROMPT.format(query=query)}],
             system="你是查询分类器，只输出 JSON。")
    classification = clean_json(r["content"])
    complexity = classification.get("complexity", "medium") if classification else "medium"

    route = ROUTING_TABLE.get(complexity, ROUTING_TABLE["medium"])
    return {"complexity": complexity, "route": route}


# 模拟路由执行
routing_results = []
for q in test_queries:
    route_info = classify_and_route(q)
    complexity = route_info["complexity"]
    route = route_info["route"]

    # 实际调用
    resp = call([{"role": "user", "content": q}],
                model=route["model"],
                max_tokens=route["max_tokens"],
                temperature=route["temperature"])

    routing_results.append({
        "query": q,
        "complexity": complexity,
        "model": route["model"].split("/")[-1] if "/" in route["model"] else route["model"],
        "tokens": resp["tokens"],
        "latency": resp["latency"],
    })
    print(f"  [{complexity}] {q[:35]}")
    print(f"    → model={route['model']}, tokens={resp['tokens']}, latency={resp['latency']:.2f}s")
    time.sleep(GAP)

# 成本对比
print("\n  成本对比（假设大模型比小模型贵 10x）:")
total_tokens = sum(r["tokens"] for r in routing_results)
print(f"    总 Token: {total_tokens}")
print(f"    如果全部用大模型: ~{total_tokens * 10} (估算)")
print(f"    路由后: ~{total_tokens} (估算)")
print(f"    节省: ~{(1 - total_tokens/(total_tokens*10))*100:.0f}%")


# ══════════════════════════════════════════════════════════
# ③ 语义缓存：相同语义直接返回缓存
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 语义缓存：相同语义直接返回缓存")
print("=" * 60)

class SemanticCache:
    """语义缓存：基于文本相似度的缓存"""

    def __init__(self, similarity_threshold=0.8):
        self.cache = {}  # {hash: {query, response, tokens, timestamp}}
        self.threshold = similarity_threshold
        self.hits = 0
        self.misses = 0

    def _text_similarity(self, a: str, b: str) -> float:
        """简单的字符级相似度（生产环境用 embedding）"""
        set_a = set(a)
        set_b = set(b)
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    def get(self, query: str) -> dict:
        """查找缓存"""
        best_match = None
        best_sim = 0.0
        for key, entry in self.cache.items():
            sim = self._text_similarity(query, entry["query"])
            if sim > best_sim:
                best_sim = sim
                best_match = entry

        if best_match and best_sim >= self.threshold:
            self.hits += 1
            return {"hit": True, "response": best_match["response"],
                    "similarity": best_sim, "original_query": best_match["query"]}
        self.misses += 1
        return {"hit": False, "response": None, "similarity": best_sim}

    def set(self, query: str, response: str, tokens: int):
        """写入缓存"""
        key = hashlib.md5(query.encode()).hexdigest()[:8]
        self.cache[key] = {
            "query": query,
            "response": response,
            "tokens": tokens,
            "timestamp": time.time(),
        }

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": hit_rate,
                "cache_size": len(self.cache)}


cache = SemanticCache(similarity_threshold=0.7)

# 测试缓存
cache_queries = [
    "客服系统支持哪些功能？",       # 新查询 → miss
    "客服系统有哪些功能？",         # 相似查询 → hit
    "产品价格是多少？",            # 新查询 → miss
    "价格是多少？",                # 相似查询 → hit
    "帮我设计一个 RAG 架构",      # 新查询 → miss
]

for q in cache_queries:
    result = cache.get(q)
    if result["hit"]:
        print(f"  🎯 缓存命中 (sim={result['similarity']:.2f}): '{q[:30]}'")
        print(f"     原始查询: '{result['original_query'][:30]}'")
    else:
        # 调用 LLM 并写入缓存
        resp = call([{"role": "user", "content": q}], max_tokens=100)
        cache.set(q, resp["content"], resp["tokens"])
        print(f"  📝 缓存未命中 (sim={result['similarity']:.2f}): '{q[:30]}' → 已写入缓存")
    time.sleep(GAP)

stats = cache.stats()
print(f"\n  缓存统计: hits={stats['hits']}, misses={stats['misses']}, "
      f"hit_rate={stats['hit_rate']:.0%}, size={stats['cache_size']}")


# ══════════════════════════════════════════════════════════
# ④ 成本对比：路由 + 缓存前后的成本
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 成本对比")
print("=" * 60)

# 模拟 100 个请求
import random
random.seed(42)

queries_pool = [
    "客服系统支持哪些功能？",
    "产品价格是多少？",
    "对比产品 A 和 B",
    "帮我设计一个客服系统",
    "库存还有多少？",
]

# 场景 1：无路由无缓存（全部用大模型）
total_tokens_no_opt = 0
for _ in range(100):
    q = random.choice(queries_pool)
    total_tokens_no_opt += 200  # 假设每次 200 tokens

# 场景 2：有路由无缓存（使用预计算的复杂度分布，不调用 LLM）
total_tokens_routed = 0
# 基于 ① 的分类结果：simple=2, medium=3, complex=0
complexity_dist = {"simple": 2, "medium": 3, "complex": 0}
total_q = sum(complexity_dist.values())
for _ in range(100):
    # 按比例随机选择复杂度
    r = random.random() * total_q
    cumsum = 0
    complexity = "medium"
    for c, count in complexity_dist.items():
        cumsum += count
        if r < cumsum:
            complexity = c
            break
    tokens = {"simple": 80, "medium": 150, "complex": 400}[complexity]
    total_tokens_routed += tokens

# 场景 3：有路由 + 缓存（假设 40% 缓存命中）
total_tokens_cached = 0
cache_hits = 0
for _ in range(100):
    if random.random() < 0.4:
        cache_hits += 1
        total_tokens_cached += 0  # 缓存命中不消耗 Token
    else:
        r = random.random() * total_q
        cumsum = 0
        complexity = "medium"
        for c, count in complexity_dist.items():
            cumsum += count
            if r < cumsum:
                complexity = c
                break
        tokens = {"simple": 80, "medium": 150, "complex": 400}[complexity]
        total_tokens_cached += tokens

print(f"  场景 1（无优化）:    {total_tokens_no_opt} tokens")
print(f"  场景 2（仅路由）:    {total_tokens_routed} tokens (节省 {(1-total_tokens_routed/total_tokens_no_opt)*100:.0f}%)")
print(f"  场景 3（路由+缓存）: {total_tokens_cached} tokens (节省 {(1-total_tokens_cached/total_tokens_no_opt)*100:.0f}%, 缓存命中 {cache_hits}%)")
print(f"\n  假设 1000 tokens = 0.001 元:")
print(f"    场景 1 成本: {total_tokens_no_opt/1000*0.001:.4f} 元")
print(f"    场景 3 成本: {total_tokens_cached/1000*0.001:.4f} 元")


# ══════════════════════════════════════════════════════════
# ⑤ 降级策略：主模型失败 → 备用模型
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑤ 降级策略")
print("=" * 60)

class ModelRouter:
    """模型路由器：支持降级"""

    def __init__(self):
        self.models = [
            {"name": "primary", "model": MODEL_SMALL, "priority": 1},
            {"name": "fallback", "model": MODEL_SMALL, "priority": 2},  # 实际应为不同模型
        ]
        self.failure_count = {}
        self.circuit_open = {}

    def call_with_fallback(self, messages, **kwargs) -> dict:
        """带降级的调用"""
        for m in sorted(self.models, key=lambda x: x["priority"]):
            model_name = m["name"]

            # 熔断检查
            if self.circuit_open.get(model_name, False):
                print(f"    ⚡ {model_name} 熔断中，跳过")
                continue

            result = call(messages, model=m["model"], **kwargs)
            if result["error"] is None:
                print(f"    ✅ {model_name} 调用成功")
                return result

            # 失败计数
            self.failure_count[model_name] = self.failure_count.get(model_name, 0) + 1
            print(f"    ❌ {model_name} 失败: {result['error'][:50]}")

            # 熔断：连续失败 3 次
            if self.failure_count[model_name] >= 3:
                self.circuit_open[model_name] = True
                print(f"    🔴 {model_name} 熔断开启")

        return {"content": "服务暂时不可用，请稍后重试", "tokens": 0, "latency": 0,
                "model": "none", "error": "all models failed"}

router = ModelRouter()

# 模拟降级
print("  模拟正常调用:")
r = router.call_with_fallback([{"role": "user", "content": "你好"}], max_tokens=50)
print(f"  响应: {r['content'][:60]}")

print("\n  降级策略总结:")
print("  1. 主模型失败 → 自动切换备用模型")
print("  2. 连续失败 → 熔断器开启（防止级联故障）")
print("  3. 全部失败 → 返回默认回复（不暴露错误）")
print("  4. 熔断恢复 → 半开状态试探")

print("\n" + "=" * 60)
print("W10 第 1 课总结")
print("=" * 60)
print("""
  ① 意图分类：判断查询复杂度 (simple/medium/complex)
  ② 模型路由：根据复杂度选择合适的模型和参数
  ③ 语义缓存：相同语义直接返回缓存（节省 Token）
  ④ 成本对比：路由 + 缓存可节省 60-80% Token
  ⑤ 降级策略：主模型 → 备用模型 → 默认回复

  企业级关键：不是所有请求都需要最强模型
  路由 + 缓存 + 降级 = 性能 + 成本 + 可靠性
""")
print("=" * 60)
print("\n  问你一个问题（引向 W10 第 2 课）：")
print("  企业级部署需要考虑哪些架构模式？")
print("  （微服务、消息队列、数据库分片……）")
print("=" * 60)
