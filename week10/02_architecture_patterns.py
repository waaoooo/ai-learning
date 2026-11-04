"""
W10 第 2 课：企业级架构模式

演示：
  ① 微服务注册与发现（Service Registry）
  ② 事件驱动架构（Event Bus）
  ③ CQRS 读写分离
  ④ 服务间通信（同步 vs 异步）
  ⑤ 弹性模式（熔断、重试、限流）
"""
import os
import json
import time
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 1.5


def call(messages, temperature=0.0, system=None, max_tokens=100) -> str:
    try:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        resp = client.chat.completions.create(
            model=MODEL, messages=msgs, temperature=temperature,
            max_tokens=max_tokens, extra_body={"thinking": {"type": "disabled"}},
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[ERROR] {e}"


# ══════════════════════════════════════════════════════════
# ① 微服务注册与发现
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 微服务注册与发现 (Service Registry)")
print("=" * 60)


@dataclass
class ServiceInstance:
    name: str
    host: str
    port: int
    health: str = "healthy"
    metadata: dict = field(default_factory=dict)


class ServiceRegistry:
    """服务注册中心：注册、发现、健康检查"""

    def __init__(self):
        self.services = defaultdict(list)  # {service_name: [instances]}

    def register(self, instance: ServiceInstance):
        self.services[instance.name].append(instance)
        print(f"  📋 注册: {instance.name} @ {instance.host}:{instance.port}")

    def discover(self, service_name: str) -> ServiceInstance:
        """服务发现：返回一个健康实例（轮询）"""
        instances = [i for i in self.services.get(service_name, []) if i.health == "healthy"]
        if not instances:
            raise Exception(f"无可用实例: {service_name}")
        # 简单轮询
        idx = hash(time.time()) % len(instances)
        return instances[idx]

    def deregister(self, service_name: str, host: str, port: int):
        self.services[service_name] = [
            i for i in self.services[service_name]
            if not (i.host == host and i.port == port)
        ]
        print(f"  🗑️ 注销: {service_name} @ {host}:{port}")

    def health_check(self, service_name: str):
        """模拟健康检查"""
        for inst in self.services.get(service_name, []):
            # 模拟：随机标记为 healthy/unhealthy
            inst.health = "healthy"
        print(f"  💚 健康检查: {service_name} → {len(self.services[service_name])} 个健康实例")


registry = ServiceRegistry()

# 注册服务
services = [
    ServiceInstance("chat-service", "10.0.1.1", 8001, metadata={"version": "v2.1"}),
    ServiceInstance("chat-service", "10.0.1.2", 8001, metadata={"version": "v2.1"}),
    ServiceInstance("rag-service", "10.0.2.1", 8002, metadata={"version": "v1.3"}),
    ServiceInstance("tool-service", "10.0.3.1", 8003, metadata={"version": "v1.0"}),
    ServiceInstance("llm-proxy", "10.0.4.1", 8004, metadata={"version": "v3.0"}),
]

for s in services:
    registry.register(s)

# 服务发现
print("\n  服务发现:")
for name in ["chat-service", "rag-service", "llm-proxy"]:
    inst = registry.discover(name)
    print(f"    {name} → {inst.host}:{inst.port} (version={inst.metadata['version']})")

registry.health_check("chat-service")


# ══════════════════════════════════════════════════════════
# ② 事件驱动架构 (Event Bus)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 事件驱动架构 (Event Bus)")
print("=" * 60)


@dataclass
class Event:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: str = ""
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""


class EventBus:
    """事件总线：发布-订阅模式"""

    def __init__(self):
        self.subscribers = defaultdict(list)  # {event_type: [handlers]}
        self.event_log = []

    def subscribe(self, event_type: str, handler: Callable):
        self.subscribers[event_type].append(handler)
        print(f"  📡 订阅: {handler.__name__} → {event_type}")

    def publish(self, event: Event):
        self.event_log.append(event)
        handlers = self.subscribers.get(event.event_type, [])
        handlers += self.subscribers.get("*", [])  # 通配符订阅
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"  ❌ Handler {handler.__name__} 失败: {e}")

    def replay(self, event_type: str = None):
        """事件重放"""
        events = [e for e in self.event_log if event_type is None or e.event_type == event_type]
        print(f"  🔄 重放 {len(events)} 个事件")
        return events


bus = EventBus()

# 定义事件处理器
def on_user_message(event: Event):
    print(f"    [ChatService] 收到消息: {event.payload.get('text', '')[:30]}")

def on_tool_invoke(event: Event):
    print(f"    [ToolService] 调用工具: {event.payload.get('tool', '')}")

def on_audit_log(event: Event):
    print(f"    [AuditService] 记录审计: {event.payload.get('action', '')}")

def on_metric_emit(event: Event):
    print(f"    [MetricsService] 指标: {event.payload.get('metric', '')}={event.payload.get('value', '')}")

# 订阅事件
bus.subscribe("user.message", on_user_message)
bus.subscribe("tool.invoke", on_tool_invoke)
bus.subscribe("audit.log", on_audit_log)
bus.subscribe("metric.emit", on_metric_emit)
bus.subscribe("*", on_audit_log)  # 通配符：审计所有事件

# 发布事件
print("\n  发布事件:")
bus.publish(Event(event_type="user.message", payload={"text": "客服系统支持哪些功能？"}, source="gateway"))
bus.publish(Event(event_type="tool.invoke", payload={"tool": "search_products", "query": "客服"}, source="chat-service"))
bus.publish(Event(event_type="metric.emit", payload={"metric": "tokens", "value": 150}, source="llm-proxy"))

print(f"\n  事件日志: {len(bus.event_log)} 个事件")


# ══════════════════════════════════════════════════════════
# ③ CQRS 读写分离
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ CQRS 读写分离")
print("=" * 60)


class WriteStore:
    """写库：处理命令（Command）"""

    def __init__(self):
        self.data = {}  # {id: record}
        self.event_queue = deque()

    def execute(self, command: str, payload: dict) -> dict:
        if command == "create_conversation":
            conv_id = f"conv-{len(self.data)+1}"
            self.data[conv_id] = {
                "id": conv_id,
                "user_id": payload["user_id"],
                "messages": [],
                "created_at": time.time(),
            }
            self.event_queue.append({"type": "conversation_created", "id": conv_id})
            return {"status": "created", "id": conv_id}

        elif command == "add_message":
            conv_id = payload["conv_id"]
            if conv_id in self.data:
                self.data[conv_id]["messages"].append(payload["message"])
                self.event_queue.append({"type": "message_added", "conv_id": conv_id})
                return {"status": "added"}
            return {"status": "not_found"}

        return {"status": "unknown_command"}


class ReadStore:
    """读库：处理查询（Query），带缓存"""

    def __init__(self):
        self.cache = {}  # {key: data}
        self.projections = {}  # {conv_id: projected_data}

    def project(self, event: dict, write_store: WriteStore):
        """从写库投影到读库"""
        if event["type"] == "conversation_created":
            conv = write_store.data[event["id"]]
            self.projections[event["id"]] = {
                "id": conv["id"],
                "user_id": conv["user_id"],
                "message_count": 0,
                "last_message": None,
            }
        elif event["type"] == "message_added":
            conv_id = event["conv_id"]
            if conv_id in self.projections:
                conv = write_store.data[conv_id]
                self.projections[conv_id]["message_count"] = len(conv["messages"])
                self.projections[conv_id]["last_message"] = conv["messages"][-1] if conv["messages"] else None

    def query(self, conv_id: str) -> dict:
        """查询（带缓存）"""
        cache_key = f"conv:{conv_id}"
        if cache_key in self.cache:
            return {"source": "cache", "data": self.cache[cache_key]}
        if conv_id in self.projections:
            data = self.projections[conv_id]
            self.cache[cache_key] = data
            return {"source": "projection", "data": data}
        return {"source": "not_found", "data": None}


# 演示 CQRS
write_store = WriteStore()
read_store = ReadStore()

# 写操作
r1 = write_store.execute("create_conversation", {"user_id": "u001"})
print(f"  写: 创建对话 → {r1}")
conv_id = r1["id"]

write_store.execute("add_message", {"conv_id": conv_id, "message": {"role": "user", "content": "你好"}})
write_store.execute("add_message", {"conv_id": conv_id, "message": {"role": "assistant", "content": "你好！"}})
print(f"  写: 添加 2 条消息")

# 投影到读库
for event in write_store.event_queue:
    read_store.project(event, write_store)

# 读操作
q1 = read_store.query(conv_id)
print(f"  读: {q1['source']} → {q1['data']}")

# 缓存命中
q2 = read_store.query(conv_id)
print(f"  读: {q2['source']} → (缓存命中)")


# ══════════════════════════════════════════════════════════
# ④ 同步 vs 异步通信
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 同步 vs 异步通信")
print("=" * 60)

# 同步调用（RPC）
def sync_call(service: str, method: str, params: dict) -> dict:
    """同步 RPC 调用：阻塞等待结果"""
    start = time.time()
    inst = registry.discover(service)
    # 模拟网络延迟
    time.sleep(0.1)
    latency = time.time() - start
    return {"result": f"{service}.{method}({params})", "latency": latency, "mode": "sync"}

# 异步调用（消息队列）
def async_call(service: str, method: str, params: dict) -> str:
    """异步调用：发布事件，不等待结果"""
    event = Event(event_type=f"{service}.{method}", payload=params, source="caller")
    bus.publish(event)
    return event.event_id

# 对比
print("  同步调用:")
r = sync_call("rag-service", "retrieve", {"query": "客服功能"})
print(f"    结果: {r['result']}, 延迟: {r['latency']*1000:.0f}ms")

print("  异步调用:")
eid = async_call("rag-service", "retrieve", {"query": "客服功能"})
print(f"    事件ID: {eid} (不等待结果，立即返回)")

print("\n  选择指南:")
print("  | 场景 | 同步 | 异步 |")
print("  |---|---|---|")
print("  | 用户等待结果 | ✅ | ❌ |")
print("  | 后台任务 | ❌ | ✅ |")
print("  | 服务解耦 | ❌ | ✅ |")
print("  | 低延迟 | ✅ | ❌ |")


# ══════════════════════════════════════════════════════════
# ⑤ 弹性模式 (Resilience Patterns)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑤ 弹性模式 (Resilience Patterns)")
print("=" * 60)


class CircuitBreaker:
    """熔断器：防止级联故障"""

    def __init__(self, failure_threshold=3, recovery_timeout=5):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "closed"  # closed/open/half-open
        self.last_failure_time = 0

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                print(f"    🔄 熔断器: half-open (试探)")
            else:
                raise Exception("熔断器开启，拒绝调用")

        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                print(f"    ✅ 熔断器: closed (恢复)")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                print(f"    🔴 熔断器: open (熔断)")
            raise e


class RateLimiter:
    """限流器：令牌桶算法"""

    def __init__(self, rate=10, capacity=20):
        self.rate = rate  # 每秒生成令牌数
        self.capacity = capacity  # 桶容量
        self.tokens = capacity
        self.last_time = time.time()

    def acquire(self) -> bool:
        now = time.time()
        elapsed = now - self.last_time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_time = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


# 演示熔断器
cb = CircuitBreaker(failure_threshold=2, recovery_timeout=2)

def unstable_service():
    raise Exception("服务不可用")

print("  熔断器演示:")
for i in range(4):
    try:
        cb.call(unstable_service)
    except Exception as e:
        print(f"    调用 {i+1}: {e} (state={cb.state}, failures={cb.failure_count})")
    time.sleep(0.5)

# 等待恢复
print("  等待恢复...")
time.sleep(2.5)
try:
    cb.call(lambda: "OK")
    print(f"    恢复后调用: 成功 (state={cb.state})")
except Exception as e:
    print(f"    恢复后调用: {e}")

# 限流器演示
print("\n  限流器演示:")
rl = RateLimiter(rate=5, capacity=10)
allowed = 0
rejected = 0
for i in range(15):
    if rl.acquire():
        allowed += 1
    else:
        rejected += 1
print(f"    15 个请求: {allowed} 通过, {rejected} 被限流")


# ══════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("W10 第 2 课总结")
print("=" * 60)
print("""
  ① 服务注册与发现：微服务定位与负载均衡
  ② 事件驱动架构：发布-订阅、解耦、可重放
  ③ CQRS：读写分离、投影、缓存
  ④ 同步 vs 异步：RPC vs 消息队列
  ⑤ 弹性模式：熔断器、限流器、重试

  企业级架构 = 微服务 + 事件驱动 + CQRS + 弹性模式
  核心原则：解耦、异步、弹性、可观测
""")
print("=" * 60)
print("\n  问你一个问题（引向 W10 第 3 课）：")
print("  企业级 Agent 系统的数据库设计应该考虑什么？")
print("  （多租户隔离、向量存储、会话管理……）")
print("=" * 60)
