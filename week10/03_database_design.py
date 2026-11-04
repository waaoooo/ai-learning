"""
W10 第 3 课：数据库设计与多租户

演示：
  ① 多租户隔离（tenant_id 行级隔离）
  ② 数据模型（用户/会话/消息/记忆/审计）
  ③ 分层存储（热/温/冷）
  ④ 索引策略
  ⑤ 软删除与审计追踪
"""
import os
import sqlite3
import json
import time
import uuid
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 1.5

DB_PATH = os.path.expanduser("~/hermes/ai-learning/week10/enterprise.db")

# 清理旧数据库
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ══════════════════════════════════════════════════════════
# ① 数据库 Schema 设计
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 数据库 Schema 设计")
print("=" * 60)

conn = get_conn()

# 租户表
conn.execute("""
CREATE TABLE tenants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    plan TEXT DEFAULT 'free',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# 用户表
conn.execute("""
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    email TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id)
)
""")

# 会话表
conn.execute("""
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
)
""")

# 消息表
conn.execute("""
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conv_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tokens INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conv_id) REFERENCES conversations(id)
)
""")

# 记忆表
conn.execute("""
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    type TEXT NOT NULL,  -- short/semantic/episodic/procedural
    content TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id)
)
""")

# 工具调用表
conn.execute("""
CREATE TABLE tool_calls (
    id TEXT PRIMARY KEY,
    conv_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    params TEXT,
    result TEXT,
    status TEXT DEFAULT 'success',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# 审计日志表
conn.execute("""
CREATE TABLE audit_logs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT,
    action TEXT NOT NULL,
    resource TEXT,
    detail TEXT,
    ip_address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# 使用指标表
conn.execute("""
CREATE TABLE usage_metrics (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    user_id TEXT,
    metric_type TEXT NOT NULL,  -- tokens/cost/latency
    value REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
print("  ✅ 8 张表创建完成")


# ══════════════════════════════════════════════════════════
# ② 索引策略
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 索引策略")
print("=" * 60)

indexes = [
    # 会话查询：按租户 + 用户 + 时间
    "CREATE INDEX idx_conv_tenant_user ON conversations(tenant_id, user_id, created_at DESC)",
    # 消息查询：按会话 + 时间
    "CREATE INDEX idx_msg_conv_time ON messages(conv_id, created_at)",
    # 租户隔离：所有表的 tenant_id 索引
    "CREATE INDEX idx_users_tenant ON users(tenant_id)",
    "CREATE INDEX idx_memories_tenant ON memories(tenant_id, user_id)",
    "CREATE INDEX idx_toolcalls_tenant ON tool_calls(tenant_id, conv_id)",
    "CREATE INDEX idx_audit_tenant ON audit_logs(tenant_id, created_at DESC)",
    "CREATE INDEX idx_usage_tenant ON usage_metrics(tenant_id, created_at DESC)",
    # 软删除：过滤已删除记录
    "CREATE INDEX idx_conv_deleted ON conversations(deleted_at) WHERE deleted_at IS NULL",
]

for idx_sql in indexes:
    conn.execute(idx_sql)
    idx_name = idx_sql.split("idx_")[1].split(" ")[0] if "idx_" in idx_sql else "index"
    print(f"  ✅ {idx_name}")

conn.commit()
print(f"  共 {len(indexes)} 个索引")


# ══════════════════════════════════════════════════════════
# ③ 多租户数据隔离演示
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 多租户数据隔离")
print("=" * 60)

# 创建两个租户
tenants = [
    ("t1", "企业A", "pro"),
    ("t2", "企业B", "free"),
]
for tid, name, plan in tenants:
    conn.execute("INSERT INTO tenants (id, name, plan) VALUES (?, ?, ?)", (tid, name, plan))

# 每个租户创建用户
users = [
    ("u1", "t1", "alice@enterprise-a.com", "admin"),
    ("u2", "t1", "bob@enterprise-a.com", "user"),
    ("u3", "t2", "charlie@enterprise-b.com", "user"),
]
for uid, tid, email, role in users:
    conn.execute("INSERT INTO users (id, tenant_id, email, role) VALUES (?, ?, ?, ?)",
                 (uid, tid, email, role))

# 创建会话（每个租户的数据独立）
convs = [
    ("c1", "t1", "u1", "产品咨询"),
    ("c2", "t1", "u2", "技术支持"),
    ("c3", "t2", "u3", "价格查询"),
]
for cid, tid, uid, title in convs:
    conn.execute("INSERT INTO conversations (id, tenant_id, user_id, title) VALUES (?, ?, ?, ?)",
                 (cid, tid, uid, title))

# 添加消息
msgs = [
    ("m1", "c1", "t1", "user", "客服系统支持哪些功能？", 15),
    ("m2", "c1", "t1", "assistant", "支持智能问答、多轮对话、知识库检索", 30),
    ("m3", "c2", "t1", "user", "如何重置密码？", 10),
    ("m4", "c3", "t2", "user", "价格是多少？", 8),
]
for mid, cid, tid, role, content, tokens in msgs:
    conn.execute("INSERT INTO messages (id, conv_id, tenant_id, role, content, tokens) VALUES (?, ?, ?, ?, ?, ?)",
                 (mid, cid, tid, role, content, tokens))

conn.commit()

# 演示租户隔离
print("\n  租户 A (t1) 的数据:")
cursor = conn.execute("SELECT id, title FROM conversations WHERE tenant_id = 't1' AND deleted_at IS NULL")
for row in cursor:
    print(f"    {row['id']}: {row['title']}")

print("\n  租户 B (t2) 的数据:")
cursor = conn.execute("SELECT id, title FROM conversations WHERE tenant_id = 't2' AND deleted_at IS NULL")
for row in cursor:
    print(f"    {row['id']}: {row['title']}")

# 隔离验证：租户 A 无法看到租户 B 的数据
print("\n  隔离验证:")
cursor = conn.execute("SELECT COUNT(*) as cnt FROM conversations WHERE tenant_id = 't1'")
print(f"    租户 A 看到的会话数: {cursor.fetchone()['cnt']} (应为 2)")
cursor = conn.execute("SELECT COUNT(*) as cnt FROM conversations WHERE tenant_id = 't2'")
print(f"    租户 B 看到的会话数: {cursor.fetchone()['cnt']} (应为 1)")

# 尝试跨租户访问（应该失败）
print("\n  跨租户访问测试:")
try:
    cursor = conn.execute("SELECT * FROM conversations WHERE id = 'c3' AND tenant_id = 't1'")
    row = cursor.fetchone()
    if row is None:
        print("    ✅ 租户 A 无法访问租户 B 的会话 c3（隔离有效）")
    else:
        print("    ❌ 隔离失败！")
except Exception as e:
    print(f"    错误: {e}")


# ══════════════════════════════════════════════════════════
# ④ 软删除与审计追踪
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 软删除与审计追踪")
print("=" * 60)

# 软删除会话 c2
conn.execute("UPDATE conversations SET deleted_at = CURRENT_TIMESTAMP WHERE id = 'c2'")
conn.commit()
print("  软删除: c2 (技术支持)")

# 查询时过滤已删除
cursor = conn.execute("SELECT id, title FROM conversations WHERE tenant_id = 't1' AND deleted_at IS NULL")
print("  租户 A 活跃会话:")
for row in cursor:
    print(f"    {row['id']}: {row['title']}")

# 审计日志记录
audit_entries = [
    ("a1", "t1", "u1", "create_conversation", "conversation", "c1", "192.168.1.1"),
    ("a2", "t1", "u1", "delete_conversation", "conversation", "c2", "192.168.1.1"),
    ("a3", "t2", "u3", "create_conversation", "conversation", "c3", "192.168.1.2"),
]
for aid, tid, uid, action, resource, detail, ip in audit_entries:
    conn.execute("""INSERT INTO audit_logs (id, tenant_id, user_id, action, resource, detail, ip_address)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                 (aid, tid, uid, action, resource, detail, ip))
conn.commit()

print("\n  审计日志:")
cursor = conn.execute("SELECT * FROM audit_logs ORDER BY created_at")
for row in cursor:
    print(f"    [{row['created_at']}] {row['user_id']} → {row['action']} on {row['resource']} ({row['detail']})")


# ══════════════════════════════════════════════════════════
# ⑤ 分层存储与数据生命周期
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑤ 分层存储与数据生命周期")
print("=" * 60)

# 模拟数据分层
storage_tiers = {
    "hot (Redis)": {
        "data": "当前会话上下文、在线状态",
        "ttl": "30 分钟",
        "size": "小",
        "speed": "极快",
    },
    "warm (PostgreSQL)": {
        "data": "完整对话历史、用户画像、记忆",
        "ttl": "30天-1年",
        "size": "中",
        "speed": "快",
    },
    "cold (S3/Glacier)": {
        "data": "归档对话、审计日志、备份",
        "ttl": "1-7年",
        "size": "大",
        "speed": "慢",
    },
}

for tier, info in storage_tiers.items():
    print(f"\n  {tier}:")
    for k, v in info.items():
        print(f"    {k}: {v}")

# 数据生命周期策略
print("\n  数据生命周期:")
lifecycle = [
    ("会话上下文", "hot", "30min", "自动过期"),
    ("对话历史", "warm", "30天", "移至 cold"),
    ("审计日志", "warm", "90天", "移至 cold"),
    ("记忆数据", "warm", "永久", "用户删除时清除"),
    ("向量数据", "warm", "永久", "文档更新时重建"),
]
for data, tier, ttl, action in lifecycle:
    print(f"    {data:12s} → {tier:6s} (TTL={ttl:8s}) → {action}")


# ══════════════════════════════════════════════════════════
# ⑥ 使用指标与计费
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑥ 使用指标与计费")
print("=" * 60)

# 记录使用指标
metrics = [
    ("t1", "u1", "tokens", 150),
    ("t1", "u1", "tokens", 200),
    ("t1", "u2", "tokens", 80),
    ("t2", "u3", "tokens", 50),
]
for tid, uid, mtype, value in metrics:
    conn.execute("INSERT INTO usage_metrics (id, tenant_id, user_id, metric_type, value) VALUES (?, ?, ?, ?, ?)",
                 (str(uuid.uuid4())[:8], tid, uid, mtype, value))
conn.commit()

# 按租户统计用量
print("  按租户 Token 用量:")
cursor = conn.execute("""
    SELECT t.name, SUM(u.value) as total_tokens, COUNT(DISTINCT u.user_id) as users
    FROM usage_metrics u JOIN tenants t ON u.tenant_id = t.id
    WHERE u.metric_type = 'tokens'
    GROUP BY t.id
""")
for row in cursor:
    cost = row["total_tokens"] / 1000 * 0.001  # 假设 1000 tokens = 0.001 元
    print(f"    {row['name']}: {row['total_tokens']} tokens, {row['users']} 用户, 成本 ≈ {cost:.4f} 元")

# 按用户统计用量
print("\n  按用户 Token 用量 (租户 A):")
cursor = conn.execute("""
    SELECT u.email, SUM(m.value) as total_tokens
    FROM usage_metrics m JOIN users u ON m.user_id = u.id
    WHERE m.tenant_id = 't1' AND m.metric_type = 'tokens'
    GROUP BY m.user_id
""")
for row in cursor:
    print(f"    {row['email']}: {row['total_tokens']} tokens")


# ══════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("W10 第 3 课总结")
print("=" * 60)
print("""
  ① Schema 设计：8 张表覆盖租户/用户/会话/消息/记忆/工具/审计/指标
  ② 索引策略：tenant_id + 时间 + 软删除，优化查询性能
  ③ 多租户隔离：行级 tenant_id 隔离，跨租户访问返回空
  ④ 软删除：deleted_at 标记，查询时过滤，可恢复
  ⑤ 分层存储：hot (Redis) / warm (PostgreSQL) / cold (S3)
  ⑥ 使用指标：按租户/用户统计 Token 用量和成本

  数据库是 Agent 系统的"记忆"，设计好坏决定系统上限
""")
print("=" * 60)
print("\n  问你一个问题（引向 W10 第 4 课）：")
print("  企业级 Agent 系统的部署流程应该是什么？")
print("  （CI/CD、容器化、监控告警……）")
print("=" * 60)

conn.close()
