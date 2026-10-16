"""
W2 第 2 步：切片策略 —— 硬切会切碎事实
对比：固定字数硬切 vs 按段落切 + 重叠
"""

# 模拟一份"公司制度"文档（每行是一条独立制度）
DOCUMENT = """差旅制度
1. 住宿标准：一线城市每晚不超过 500 元，其他城市不超过 350 元，超出部分自理。
2. 交通标准：高铁限二等座，飞机限经济舱，打车需事前审批。
3. 餐补标准：出差期间每人每天 100 元，凭发票实报实销。
4. 报销时限：费用发生后 30 天内提交，逾期不予报销。"""


def naive_split(text, size=30):
    """策略1：固定字数硬切（不关心语义）"""
    return [text[i:i + size] for i in range(0, len(text), size)]


def paragraph_split(text, size=60, overlap=15):
    """策略2：按段落切 + 重叠（保住语义完整性）"""
    lines = [l for l in text.split("\n") if l.strip()]
    chunks, buf = [], ""
    for line in lines:
        if len(buf) + len(line) > size and buf:
            chunks.append(buf)
            # 重叠：保留上一段结尾，切断层时信息不丢
            buf = buf[-overlap:] + "\n" + line
        else:
            buf = (buf + "\n" + line).strip()
    if buf:
        chunks.append(buf)
    return chunks


print("=" * 60)
print("【策略1】固定字数硬切（size=30）")
print("=" * 60)
for i, c in enumerate(naive_split(DOCUMENT), 1):
    print(f"  chunk{i}: {c!r}")
print("  ↑ 看 chunk1 结尾的 '其' 和 chunk2 开头的 '他城市' —— '其他城市' 被切成两半了")

print()
print("=" * 60)
print("【策略2】按段落切 + 重叠（size=60, overlap=15）")
print("=" * 60)
for i, c in enumerate(paragraph_split(DOCUMENT), 1):
    print(f"  chunk{i}: {c!r}")
print("  ↑ 每条制度都完整，且块与块之间有重叠，跨块的信息也能被检索到")
