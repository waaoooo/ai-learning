"""
W9 第 3 课：权限控制（RBAC + 工具白名单）

演示：
  ① RBAC 角色定义（admin / user / guest）
  ② 工具白名单（不同角色可调用不同工具）
  ③ 参数校验（工具参数必须符合 schema）
  ④ 危险操作需人工确认
  ⑤ 审计日志（记录所有工具调用）
"""
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"),
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                timeout=30, max_retries=0)
MODEL = "glm-4-flash"
GAP = 3.0


# ══════════════════════════════════════════════════════════
# ① RBAC 角色定义
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① RBAC 角色定义")
print("=" * 60)

ROLES = {
    "admin": {
        "description": "管理员：全权限",
        "allowed_tools": {"search_products", "get_product_info", "check_inventory",
                          "update_product", "delete_product", "export_data", "manage_users"},
        "dangerous_tools": {"delete_product", "manage_users"},
        "rate_limit": 100,  # 每分钟调用次数
    },
    "user": {
        "description": "普通用户：只读 + 搜索",
        "allowed_tools": {"search_products", "get_product_info", "check_inventory"},
        "dangerous_tools": set(),
        "rate_limit": 20,
    },
    "guest": {
        "description": "访客：仅搜索",
        "allowed_tools": {"search_products"},
        "dangerous_tools": set(),
        "rate_limit": 5,
    },
}

for role, info in ROLES.items():
    print(f"\n  {role} ({info['description']})")
    print(f"    允许工具: {info['allowed_tools']}")
    print(f"    危险工具: {info['dangerous_tools'] or '无'}")
    print(f"    速率限制: {info['rate_limit']}/min")


# ══════════════════════════════════════════════════════════
# ② 工具白名单 + 参数校验
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② 工具白名单 + 参数校验")
print("=" * 60)

# 工具参数 schema
TOOL_SCHEMAS = {
    "search_products": {
        "required": ["query"],
        "params": {"query": str, "limit": int},
    },
    "get_product_info": {
        "required": ["product_id"],
        "params": {"product_id": str},
    },
    "delete_product": {
        "required": ["product_id", "reason"],
        "params": {"product_id": str, "reason": str},
    },
    "manage_users": {
        "required": ["action", "user_id"],
        "params": {"action": str, "user_id": str},
    },
}


def validate_params(tool_name: str, params: dict) -> tuple[bool, str]:
    """校验工具参数是否符合 schema"""
    schema = TOOL_SCHEMAS.get(tool_name)
    if not schema:
        return False, f"未知工具: {tool_name}"

    # 检查必填参数
    for req in schema["required"]:
        if req not in params:
            return False, f"缺少必填参数: {req}"

    # 检查参数类型
    for key, val in params.items():
        if key in schema["params"]:
            expected_type = schema["params"][key]
            if not isinstance(val, expected_type):
                return False, f"参数 {key} 类型错误: 期望 {expected_type.__name__}, 实际 {type(val).__name__}"

    return True, "OK"


# 模拟工具执行
def mock_tool(tool_name: str, params: dict) -> str:
    if tool_name == "search_products":
        return f"找到 3 个匹配 '{params['query']}' 的产品"
    elif tool_name == "get_product_info":
        return f"产品 {params['product_id']}: 智能客服系统，支持多轮对话"
    elif tool_name == "delete_product":
        return f"已删除产品 {params['product_id']}，原因: {params['reason']}"
    elif tool_name == "manage_users":
        return f"已对用户 {params['user_id']} 执行 {params['action']}"
    return "未知工具"


# ══════════════════════════════════════════════════════════
# ③ 权限检查器
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 权限检查器")
print("=" * 60)


class PermissionChecker:
    def __init__(self):
        self.audit_log = []
        self.rate_counters = {}

    def check(self, user_role: str, tool_name: str, params: dict,
              user_confirmed: bool = False) -> dict:
        """检查权限，返回 {allowed, reason, need_confirmation}"""
        role = ROLES.get(user_role)
        if not role:
            return {"allowed": False, "reason": f"未知角色: {user_role}", "need_confirmation": False}

        # 1. 白名单检查
        if tool_name not in role["allowed_tools"]:
            self._audit(user_role, tool_name, params, "denied", "not in whitelist")
            return {"allowed": False, "reason": f"角色 {user_role} 无权调用 {tool_name}", "need_confirmation": False}

        # 2. 参数校验
        ok, msg = validate_params(tool_name, params)
        if not ok:
            self._audit(user_role, tool_name, params, "denied", f"param error: {msg}")
            return {"allowed": False, "reason": msg, "need_confirmation": False}

        # 3. 危险操作需人工确认
        if tool_name in role["dangerous_tools"] and not user_confirmed:
            self._audit(user_role, tool_name, params, "pending", "needs confirmation")
            return {"allowed": False, "reason": "此操作需要人工确认", "need_confirmation": True}

        # 4. 速率限制
        if not self._check_rate_limit(user_role, role["rate_limit"]):
            self._audit(user_role, tool_name, params, "denied", "rate limited")
            return {"allowed": False, "reason": "调用频率超限", "need_confirmation": False}

        # 通过所有检查
        self._audit(user_role, tool_name, params, "allowed", "OK")
        return {"allowed": True, "reason": "OK", "need_confirmation": False}

    def _check_rate_limit(self, user_role: str, limit: int) -> bool:
        now = time.time()
        if user_role not in self.rate_counters:
            self.rate_counters[user_role] = []
        # 清理 1 分钟前的记录
        self.rate_counters[user_role] = [t for t in self.rate_counters[user_role] if now - t < 60]
        if len(self.rate_counters[user_role]) >= limit:
            return False
        self.rate_counters[user_role].append(now)
        return True

    def _audit(self, user_role, tool_name, params, status, reason):
        self.audit_log.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "user_role": user_role,
            "tool": tool_name,
            "params": params,
            "status": status,
            "reason": reason,
        })

    def execute(self, user_role: str, tool_name: str, params: dict,
                user_confirmed: bool = False) -> str:
        """检查权限 + 执行工具"""
        check = self.check(user_role, tool_name, params, user_confirmed)
        if not check["allowed"]:
            if check["need_confirmation"]:
                return f"⏳ 需要人工确认: {check['reason']}"
            return f"❌ 拒绝: {check['reason']}"
        return f"✅ 执行: {mock_tool(tool_name, params)}"

    def show_audit_log(self):
        print("\n  审计日志:")
        for entry in self.audit_log:
            icon = {"allowed": "✅", "denied": "❌", "pending": "⏳"}.get(entry["status"], "📝")
            print(f"    {icon} [{entry['ts']}] {entry['user_role']} → {entry['tool']} → {entry['status']} ({entry['reason']})")


checker = PermissionChecker()


# ══════════════════════════════════════════════════════════
# ④ 实战：不同角色调用不同工具
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 实战：不同角色调用不同工具")
print("=" * 60)

test_cases = [
    # (角色, 工具, 参数, 是否确认)
    ("guest", "search_products", {"query": "客服系统"}, False),
    ("guest", "get_product_info", {"product_id": "P001"}, False),      # guest 无权
    ("user", "search_products", {"query": "客服系统"}, False),
    ("user", "get_product_info", {"product_id": "P001"}, False),
    ("user", "delete_product", {"product_id": "P001", "reason": "测试"}, False),  # user 无权
    ("admin", "search_products", {"query": "客服系统"}, False),
    ("admin", "delete_product", {"product_id": "P001", "reason": "过期"}, False),  # 需确认
    ("admin", "delete_product", {"product_id": "P001", "reason": "过期"}, True),   # 已确认
    ("admin", "manage_users", {"action": "ban", "user_id": "u123"}, False),        # 需确认
    ("admin", "manage_users", {"action": "ban", "user_id": "u123"}, True),         # 已确认
    # 参数错误
    ("user", "search_products", {}, False),                                        # 缺少 query
    ("user", "get_product_info", {"product_id": 123}, False),                       # 类型错误
]

for role, tool, params, confirmed in test_cases:
    result = checker.execute(role, tool, params, confirmed)
    print(f"\n  [{role}] {tool}({params})")
    print(f"    → {result}")

checker.show_audit_log()

# ══════════════════════════════════════════════════════════
# ⑤ 总结
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑤ 总结：权限控制的 5 层")
print("=" * 60)
print("""
  1. RBAC 角色定义 —— 不同角色不同权限
  2. 工具白名单 —— 只允许调用特定工具
  3. 参数校验 —— 工具参数必须符合 schema
  4. 危险操作确认 —— 删除/修改需人工确认
  5. 审计日志 —— 所有调用可追溯

  关键原则：
  - 最小权限：Agent 只能调用必要工具
  - 失败安全：出错时默认拒绝
  - 审计一切：所有操作可追溯
""")
print("=" * 60)
print("\n  问你一个问题（引向 W9 第 4 课）：")
print("  除了权限控制，内容审核还需要检测什么？")
print("  （敏感词、PII、合规、毒性……）")
print("  你的想法？")
print("=" * 60)
