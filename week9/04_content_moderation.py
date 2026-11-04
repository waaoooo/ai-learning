"""
W9 第 4 课：内容审核与合规

演示：
  ① 敏感词检测（黑名单 + 模糊匹配）
  ② PII 检测（手机号、邮箱、身份证号）
  ③ 合规检查（广告法、医疗合规）
  ④ 毒性检测（LLM-as-Judge）
  ⑤ 综合内容审核管道
"""
import os
import re
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


def call(messages, temperature=0.0, system=None) -> str:
    last_err: Exception = RuntimeError("call failed")
    for attempt in range(3):
        try:
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            resp = client.chat.completions.create(
                model=MODEL, messages=msgs, temperature=temperature,
                max_tokens=200, extra_body={"thinking": {"type": "disabled"}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if "contentFilter" in str(e) or "1301" in str(e):
                return "[CONTENT_FILTERED]"
            time.sleep(2 ** attempt)
    return f"[ERROR] {last_err}"


# ══════════════════════════════════════════════════════════
# ① 敏感词检测
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("① 敏感词检测")
print("=" * 60)

SENSITIVE_WORDS = {
    "政治": ["习近平", "共产党", "政府"],
    "暴力": ["杀人", "爆炸", "武器"],
    "色情": ["色情", "裸体", "性交"],
    "诈骗": ["中奖", "转账", "验证码", "密码"],
}

def check_sensitive_words(text: str) -> dict:
    """检测文本中的敏感词"""
    found = {}
    for category, words in SENSITIVE_WORDS.items():
        matched = [w for w in words if w in text]
        if matched:
            found[category] = matched
    return found

test_texts = [
    "我们的产品支持数据分析和可视化",
    "恭喜您中奖了，请转账到指定账户",
    "请提供您的验证码和密码",
    "习近平总书记强调要推进科技创新",
]

for text in test_texts:
    result = check_sensitive_words(text)
    if result:
        print(f"  ❌ '{text[:40]}' → 敏感词: {result}")
    else:
        print(f"  ✅ '{text[:40]}' → 无敏感词")


# ══════════════════════════════════════════════════════════
# ② PII 检测（个人身份信息）
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("② PII 检测（个人身份信息）")
print("=" * 60)

PII_PATTERNS = {
    "手机号": r"1[3-9]\d{9}",
    "邮箱": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "身份证号": r"\d{17}[\dXx]",
    "银行卡号": r"\d{16,19}",
    "IP地址": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
}

def detect_pii(text: str) -> dict:
    """检测文本中的 PII"""
    found = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[pii_type] = matches
    return found

pii_texts = [
    "我的手机号是 13812345678，邮箱是 test@example.com",
    "身份证号 110101199001011234，银行卡 6222021234567890123",
    "服务器 IP 是 192.168.1.100",
    "今天天气不错，适合出门",
]

for text in pii_texts:
    result = detect_pii(text)
    if result:
        print(f"  ❌ '{text[:40]}' → PII: {result}")
    else:
        print(f"  ✅ '{text[:40]}' → 无 PII")


# ══════════════════════════════════════════════════════════
# ③ 合规检查（广告法、医疗合规）
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("③ 合规检查")
print("=" * 60)

COMPLIANCE_RULES = {
    "广告法违禁词": ["最", "第一", "顶级", "国家级", "独家", "最佳"],
    "医疗合规": ["治愈", "根治", "疗效", "药到病除", "无副作用"],
    "金融合规": ["保本", "稳赚", "零风险", "高收益", "翻倍"],
}

def check_compliance(text: str) -> dict:
    """合规检查"""
    violations = {}
    for category, words in COMPLIANCE_RULES.items():
        matched = [w for w in words if w in text]
        if matched:
            violations[category] = matched
    return violations

compliance_texts = [
    "我们的产品是业界最好的，效果第一",
    "这款药可以治愈所有疾病，无副作用",
    "投资稳赚不赔，零风险高收益",
    "我们的客服系统支持多轮对话和知识库检索",
]

for text in compliance_texts:
    result = check_compliance(text)
    if result:
        print(f"  ❌ '{text[:40]}' → 违规: {result}")
    else:
        print(f"  ✅ '{text[:40]}' → 合规")


# ══════════════════════════════════════════════════════════
# ④ 毒性检测（LLM-as-Judge）
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("④ 毒性检测（LLM-as-Judge）")
print("=" * 60)

TOXICITY_PROMPT = """你是一个内容审核员。判断以下文本是否包含不当内容。
文本：{text}

审核维度：
1. 毒性/侮辱性语言
2. 歧视性内容
3. 虚假信息
4. 不当引导

输出 JSON：{{"toxic": true/false, "reason": "理由", "severity": "low/medium/high"}}"""

toxic_texts = [
    "你这个白痴，什么都不会",
    "所有XX人都是骗子",
    "点击链接领取免费iPhone",
    "我们的产品支持多轮对话和知识库检索",
]

for text in toxic_texts:
    r = call([{"role": "user", "content": TOXICITY_PROMPT.format(text=text)}],
             system="你是严格的内容审核员，只输出 JSON。")
    try:
        import json as _json
        result = _json.loads(r)
        if result.get("toxic"):
            print(f"  ❌ '{text[:30]}' → 毒性: {result['reason']} (severity={result['severity']})")
        else:
            print(f"  ✅ '{text[:30]}' → 无毒性")
    except (json.JSONDecodeError, KeyError):
        print(f"  ⚠️ '{text[:30]}' → 审核结果解析失败: {r[:60]}")
    time.sleep(GAP)


# ══════════════════════════════════════════════════════════
# ⑤ 综合内容审核管道
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⑤ 综合内容审核管道")
print("=" * 60)


class ContentModerator:
    """综合内容审核器：多层检测"""

    def __init__(self):
        self.audit_log = []

    def moderate(self, text: str) -> dict:
        """综合审核，返回 {passed, violations, severity}"""
        violations = []
        severity = "low"

        # Layer 1: 敏感词
        sw = check_sensitive_words(text)
        if sw:
            violations.append({"type": "sensitive_word", "detail": sw})
            severity = "high"

        # Layer 2: PII
        pii = detect_pii(text)
        if pii:
            violations.append({"type": "pii", "detail": pii})
            severity = max(severity, "medium", key=lambda x: ["low", "medium", "high"].index(x))

        # Layer 3: 合规
        comp = check_compliance(text)
        if comp:
            violations.append({"type": "compliance", "detail": comp})
            severity = max(severity, "medium", key=lambda x: ["low", "medium", "high"].index(x))

        # Layer 4: 毒性（LLM-as-Judge）
        r = call([{"role": "user", "content": TOXICITY_PROMPT.format(text=text)}],
                 system="你是严格的内容审核员，只输出 JSON。")
        try:
            tox = json.loads(r)
            if tox.get("toxic"):
                violations.append({"type": "toxicity", "detail": tox["reason"]})
                severity = max(severity, tox.get("severity", "medium"),
                             key=lambda x: ["low", "medium", "high"].index(x))
        except (json.JSONDecodeError, KeyError):
            pass

        passed = len(violations) == 0
        self.audit_log.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "text": text[:50],
            "passed": passed,
            "violations": violations,
            "severity": severity,
        })

        return {"passed": passed, "violations": violations, "severity": severity}


moderator = ContentModerator()

test_cases = [
    "我们的客服系统支持多轮对话和知识库检索",
    "恭喜您中奖了，请转账到指定账户，提供验证码",
    "我的手机号是 13812345678",
    "我们的产品是业界最好的，效果第一",
    "你这个白痴，什么都不会",
]

for text in test_cases:
    result = moderator.moderate(text)
    icon = "✅" if result["passed"] else "❌"
    print(f"\n  {icon} '{text[:40]}'")
    print(f"     passed={result['passed']}, severity={result['severity']}")
    if result["violations"]:
        for v in result["violations"]:
            print(f"     → {v['type']}: {v['detail']}")

# 审核报告
print("\n" + "=" * 60)
print("审核报告")
print("=" * 60)
total = len(moderator.audit_log)
passed = sum(1 for e in moderator.audit_log if e["passed"])
print(f"  总审核: {total} 条")
print(f"  通过: {passed} 条")
print(f"  拒绝: {total - passed} 条")
print(f"  拒绝率: {(total-passed)/total:.0%}")

high_sev = [e for e in moderator.audit_log if e["severity"] == "high"]
print(f"  高危: {len(high_sev)} 条")
print("=" * 60)

print("\n  关键设计要点：")
print("  1. 多层检测：敏感词 → PII → 合规 → 毒性")
print("  2. 每层独立：一层失败不影响其他层")
print("  3. 严重程度分级：low/medium/high")
print("  4. 审计日志：所有审核可追溯")
print("  5. LLM-as-Judge：语义级别的毒性检测")
print("=" * 60)
print("\n  问你一个问题（引向 W9 第 5 课）：")
print("  把这四层防护（注入防御/权限/内容审核/合规）")
print("  整合成一个**安全防护体系**，你觉得架构应该长什么样？")
print("=" * 60)
