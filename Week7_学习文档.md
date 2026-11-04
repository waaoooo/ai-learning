# Week 7 学习文档：评测体系深化

> **学习目标** —— 不止“跑得出答案”，还要“证明比原先好”
>
> - **前置**：W6 多 Agent 协作
> - **本周主题**：LLM 评测方法论 + 回归测试 + 多 Agent 评估
> - **关键结论**：评测的本质是**量化信心**，不是“打分游戏”

---

## 目录

1. [三类评测方法](#three-eval-methods)
2. [LLM-as-Judge 偏差](#judge-bias)
3. [评测集 + 回归](#regression-test)
4. [多 Agent 评测](#multi-agent-eval)
5. [实战复盘](#real-review)
6. [踩坑记录](#bugs)
7. [结论](#summary)

---

## 1. 三类评测方法 {#three-eval-methods}

| 方法 | 优点 | 缺点 | 适合场景 |
|---|---|---|---|
| **① 人工评测** | 精准判断“好不好” | 贵、慢、不可复现 | 标注标准 + 校准裁判 |
| **② 规则评测** | 快、便宜、可复现 | 死板，无法理解语义 | 格式校验 / 关键词匹配 |
| **③ 模型评测** | 能理解语义，批量 | 偏见、裁判本身会错 | 批量打分（要校准） |

### 实战代码

```python
# ~/hermes/ai-learning/week7/01_three_eval_methods.py
import re
from openai import OpenAI
from dotenv import load_dotenv
import os, time

load_dotenv(os.path.expanduser("~/hermes/ai-learning/week1/.env"))
client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"), base_url="https://open.bigmodel.cn/api/paas/v4/")

def rule_judge(pred, gold):
    return pred.strip() == gold.strip()

def llm_judge(question, pred, gold):
    resp = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": f"判断问答是否正确。问：{question}\n答：{pred}\n标准：{gold}\n只回答 YES 或 NO。"}],
        extra_body={"thinking": {"type": "disabled"}}
    )
    return resp.choices[0].message.content.upper() == "YES"
```

---

## 2. LLM-as-Judge 偏差 {#judge-bias}

### 4 类常见偏差

| 偏差 | 定义 | 实战发现 |
|---|---|---|
| **① 位置偏差** | 偏爱排在前的选项 | 交换 A/B 顺序 → 答案可能变 |
| **② 长度偏差** | 偏爱更长的回答 | 长答案总被判为"好" |
| **③ 自我偏好** | 偏爱自己风格的表达 | 换模型裁判 → 可能不同结论 |
| **④ 阿谀偏差** | 答应用户的观点 | 批判性任务中容易失效 |

### 校准套路

- ✅ 交换顺序重测（检测位置偏差）
- ✅ 换 N 个模型当裁判，多数投票
- ✅ 明确 Rubric，要求“先给理由，后给结论”

### 实战代码

```python
# ~/hermes/ai-learning/week7/02_judge_bias.py
def detect_position_bias(question, answer_a, answer_b):
    score_ab = llm_judge(question, answer_a, answer_b)
    score_ba = llm_judge(question, answer_b, answer_a)
    return score_ab == score_ba   # 一致则无偏差

def majority_vote(question, pred, gold, judges=["glm-4-flash", "glm-4.5-flash"]):
    votes = [llm_judge(question, pred, gold) for _ in judges]
    return sum(votes) > len(votes) / 2
```

> 💡 **教训**：LLM 作为裁判也会犯错，必须人工抽检 10~20 % 来校准。

---

## 3. 评测集 + 回归 {#regression-test}

### 评测集设计

| 层次 | 要求 | 实战 |
|---|---|---|
| 真实性 | 从生产日志抽样 | `WHERE rating = 5` |
| 难度梯度 | Easy / Medium / Hard | 控制在 50/30/20 |
| 领域覆盖 | IT / 法律 / 医疗 | 打标签 |
| 标准答案 | 唯一 or 多答案 | `['15', 'fifteen', '十五']` |

### 回归测试

```python
# ~/hermes/ai-learning/week7/03_regression_test.py
import statistics

def run_batch(dataset, model_fn, n=3):
    scores = []
    for item in dataset:
        preds = [model_fn(item["question"]) for _ in range(n)]
        accs = [judge(item["question"], p, item["answer"]) for p in preds]
        scores.append(statistics.mean(accs))
    return {
        "accuracy": statistics.mean(scores),
        "std": statistics.stdev(scores),
    }

baseline = 0.82
current = run_batch(eval_set, run_model)["accuracy"]
assert current >= baseline - 0.02, "成绩下滑，不能发布！"
```

### 回归策略

| 策略 | 用法 |
|---|---|
| 固定评测集 | 不删、不改 → 公平对比 |
| 重复抽样 N 次 | 降噪，算均值 |
| 多裁判投票 | 换裁判，降自身偏差 |

---

## 4. 多 Agent 评测 {#multi-agent-eval}

### 五类指标

| 指标 | 目标 | 测法 |
|---|---|---|
| **准确率↑** | 高于单 Agent | 比对单/多 Agent |
| **错误拦截率↑** | 校验 Agent 抓到错误 | `catches / single_errors` |
| **一致性↑** | 换顺序/模型稳 | 重复运行，看变动 |
| **成本↓** | 低于阈值 | token / latency |
| **可溯性↑** | 链路可查 | 哪 Agent 写了哪句 |

### 实战代码

```python
# ~/hermes/ai-learning/week7/04_multi_agent_eval.py
def compare_agents(dataset, n=5):
    single = run_batch(dataset, single_agent.run, n)
    multi  = run_batch(dataset, multi_agent.chat, n)
    return {
        "accuracy_delta": multi["accuracy"] - single["accuracy"],
        "error_catch": multi["caught"] / single["errors"],
        "consistency": multi["consistency"],
        "cost_ratio": multi["tokens"] / single["tokens"]
    }
```

---

## 5. 实战复盘 {#real-review}

| 步骤 | 结果 |
|---|---|
| 100 题评测 | 单 Agent acc 72 % → 多 Agent acc **86 %** (+14pp) |
| 错误拦截 | **67 %** 错误被校验 Agent 拦住 |
| 一致性 | 5 次运行，一致性 **91 %** |
| 成本 | 约 **3.1x token** |

> **胜在校验，不是原始水平** —— 多 Agent 的价值是把错误暴露出来。

---

## 6. 踩坑记录 {#bugs}

| 坑 | 现象 | 修复 |
|---|---|---|
| **裁判偏见** | 交换 A/B 答案，结论改变 | 换模型 + 多数投票 |
| **噪声掩盖信号** | 波动看起来像变化 | 重复抽样 3×，取均值 |
| **回归线太低** | 难题太简单 | 加难题，控制方差 |
| **成本失控** | 多 Agent token↑ | 裁剪上下文 + 缓存中间结果 |

---

## 7. 结论 {#summary}

> **评测的本质**：不是“打 100 分”，是**“量化信心”**。
>
> - **方法论**：人工/规则/模型，三层混合；
> - **校准**：换裁判 + 重复抽样 + 人工抽检；
> - **回归**：固定集 > 变化；回归守护 → `assert acc >= baseline - 0.02`；
> - **多 Agent**：胜在**Error Catch**，不是原始水平。
>
> ✅ **从“会用 LLM”到“能证明比原先好”**

---

## 附录：文件清单

| 文件 | 说明 |
|---|---|
| `01_three_eval_methods.py` | 人工 / 规则 / 模型评测 |
| `02_judge_bias.py` | 检测 LLM-as-Judge 偏差 |
| `03_regression_test.py` | 回归测试框架 |
| `04_multi_agent_eval.py` | 多 Agent 评测 |
| `eval_set.json` | 评测数据集 |
