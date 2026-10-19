"""
W4 第 4 课：失败重试与策略调整（Agent 的韧性）

三级应对：
  第1级 重试     —— 同样的方法再来一次（应对偶发失败）
  第2级 换策略   —— 换个工具/方法（应对持续失败）
  第3级 优雅降级 —— 老实告诉用户做不到（应对彻底失败）
"""
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)


# ══════════════════════════════════════════════════════════
# 模拟不稳定的工具：前 2 次失败，第 3 次成功
# ══════════════════════════════════════════════════════════
class UnstableWeather:
    def __init__(self, fail_times=2):
        self.fail_times = fail_times
        self.calls = 0

    def query(self, city):
        self.calls += 1
        if self.calls <= self.fail_times:
            return f"ERROR: 服务超时（第 {self.calls} 次调用）"
        return f"{city}天气：晴，22℃"


def backup_weather(city):
    """备用数据源"""
    return f"{city}天气（备用源）：多云，20℃"


# ══════════════════════════════════════════════════════════
# 第 1 级：带重试的调用
# ══════════════════════════════════════════════════════════
def call_with_retry(fn, max_retries=3, backoff=1):
    for i in range(1, max_retries + 1):
        result = fn()
        if not result.startswith("ERROR"):
            return result, i
        print(f"     第 {i} 次失败: {result}")
        if i < max_retries:
            time.sleep(backoff * (2 ** (i - 1)))
    return None, max_retries


print("=" * 60)
print("【第 1 级】重试 —— 工具偶发失败")
print("=" * 60)
uw = UnstableWeather(fail_times=2)
print("\n调用不稳定工具（前 2 次会失败）...")
result, attempts = call_with_retry(lambda: uw.query("北京"), max_retries=4)
print(f"\n结果: {result}")
print(f"共尝试 {attempts} 次 → 重试奏效 ✅")


# ══════════════════════════════════════════════════════════
# 第 2 级：换策略 —— 主工具持续失败，切备用
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【第 2 级】换策略 —— 主工具一直失败，切备用源")
print("=" * 60)
uw2 = UnstableWeather(fail_times=99)   # 永远失败
print("\n主工具持续失败（重试 2 次）...")
result2, attempts2 = call_with_retry(lambda: uw2.query("北京"), max_retries=2)

if result2 is None:
    print("     主工具不可用 → 切换到备用数据源")
    result2 = backup_weather("北京")
    print(f"     备用源结果: {result2}")
    print(f"\n结果: {result2}")
    print("换策略奏效 ✅")


# ══════════════════════════════════════════════════════════
# 第 3 级：优雅降级 —— 所有手段都失败
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【第 3 级】优雅降级 —— 彻底失败时不硬撑")
print("=" * 60)
print("\n主工具和备用源都不可用...")
print("最终输出: 「抱歉，天气服务暂时不可用，请稍后再试。」")
print("优雅降级 ✅")


print("\n" + "=" * 60)
print("三级应对：重试 → 换策略 → 优雅降级。")
print("核心：失败不是终点，'怎么应对失败'才是 Agent 的能力。")
print("=" * 60)
