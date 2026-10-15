# Week 1 学习文档：LLM API 集成 + Prompt Engineering 基础

> **学习日期**：2026-10-14 · **语言**：Python 3.9 · **模型**：智谱 GLM-4.7-Flash（免费）
> **文档性质**：过程性学习记录，可独立阅读，无需对照聊天记录
> **前置要求**：会 Python，会跑命令行

---

## 一、本周目标与达成情况

**一句话目标**：从零跑通一次真实的 LLM API 调用，掌握 Prompt 工程、结构化输出、多轮对话、记忆系统，最终产出一个"带记忆的对话 API"。

| 任务 | 主题 | 状态 | 关键产出文件 |
|---|---|:---:|---|
| Task 1 | 真实 API 调用 + temperature + token | ✅ | `01_hello_glm.py` |
| Task 2 | System Prompt + 软/硬约束 | ✅ | `06/07/08/09_*.py` |
| Task 3 | 结构化输出（JSON Mode / Function Calling） | ✅ | `10/11_*.py` |
| Task 4 | 多轮对话 + 滑动窗口 + 记忆 | ✅ | `12/13/14/15_*.py` |
| Task 5 | 综合 API 服务 | ✅ | `16_task5_api.py` |

**最终验收**：一个可运行、可测试的智能客服 API，多轮记忆验证通过。

---

## 二、环境搭建

### 2.1 为什么选智谱 GLM

| 模型 | 说明 |
|---|---|
| `glm-4.7-flash` | ✅ 免费、能力强、推理模型（带 thinking） |
| `glm-4-flash` | ✅ 免费、非推理模型（更快更稳，兜底用） |
| `glm-4.6` / `glm-4.5` 等 | ❌ 付费，你的账号余额不足（429） |

### 2.2 关键配置（本项目通用）

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 本项目所有调用都用这个模型 + 关掉"思考"
MODEL = "glm-4.7-flash"
extra_body = {"thinking": {"type": "disabled"}}
```

> **铁律**：API Key 永远写进 `.env`，**绝不贴进聊天或硬编码**。`.env` 已在 `.gitignore` 中。

### 2.3 探测你的 Key 能用哪些模型

免费模型不在官方 `/models` 列表里，必须实测。探测脚本 `00_check_zhipu.py` 的真实输出：

```
1) /models 接口返回：
  glm-4.5, glm-4.5-air, glm-4.6, glm-4.7, glm-5, glm-5-turbo, glm-5.1, glm-5.2, glm-5.3, glm-5.3-flash

2) 逐个试候选模型：
  ✅ glm-4.7-flash  ->  （空）
  ✅ glm-4.5-flash  ->  （空）
  ✅ glm-4-flash    ->  可用
  ❌ glm-4.6        ->  429 余额不足或无可用资源包
  ❌ glm-4.5        ->  429 余额不足或无可用资源包
  ❌ glm-4.5-air    ->  429 余额不足或无可用资源包
```

**两个重要发现**：① 付费模型全部 429（免费账号只能用 flash）；② `glm-4.7-flash` 输出**空字符串` —— 这正是 3.3 节要讲的推理模型坑。

---

## 三、知识点详解

### 3.1 LLM API 调用的本质

**一句话**：LLM 调用就是一次 HTTP POST 请求，没有魔法、没有本地模型。

```
POST https://open.bigmodel.cn/api/paas/v4/chat/completions
Authorization: Bearer <你的Key>
{
  "model": "glm-4.7-flash",
  "messages": [{"role": "user", "content": "你好"}]
}
```

### 3.2 Temperature（采样随机性）

**它是什么**：控制模型选词时"多大胆"。
- `0` = 每次都选概率最高的词 → **可复现、严谨**
- 调高 = 给低概率词更多机会 → **发散、有创意，但也更容易出错**

**关键澄清**：temperature **不改变模型的"智力"**，只改变**采样随机性**。0.1 不是"更聪明"，只是"更确定"。

**真实数据**（`01_hello_glm.py`，同一问题跑不同值）：

```
temperature=0.1: 春回大地，万物复苏，生机勃勃。
temperature=0.9: 春回大地，万物复苏，充满希望与生机。
```

> ⚠️ **反直觉结论**：任务越简单，temperature 越看不出差别。只有在开放创作（写诗、起标题）时差异才明显。写代码/抽数据时用低值（0.1），创意写作时用高值（0.8-0.9）。

### 3.3 推理模型（reasoning model）与 thinking

GLM-4.5/4.7 是**推理模型**：它会先输出一段**内部思考**（`reasoning_content`），再输出**正文**（`content`）。

**翻车现场**：第一次调用给了 `max_tokens=500`，结果 `content` 为空 —— 看用量：

```
completion_tokens=500, reasoning_tokens=500   ← 500 个 token 全被"思考"吃光了
```

**实测三种应对方式**（`probe_thinking.py` 真实输出）：

| 方案 | content | reasoning | total_tokens |
|---|---|---|---|
| A) 推理模型 + **关思考** | ✅ 正常 | None | **24** |
| B) 推理模型 + 开思考 + 大额度 | ✅ 正常 | 有（755） | **782** |
| C) 非推理 `glm-4-flash` | ✅ 正常 | None | **20** |

> 💡 **成本控制第一课**：同一句话，B 比 A 贵 **30 倍**。写文案、做分类、抽字段 → 关思考又快又便宜；只有复杂推理（数学、多步规划）才值得开。

### 3.4 System Prompt（系统提示词）

**它是什么**：每次调用都**自动塞进去**的隐形指令，优先级高于用户消息。用来设定角色、约束、输出格式。

**实测效果**（`06_task2_system_prompt.py`，同一问题"什么是机器学习"）：

| System Prompt | 输出风格 |
|---|---|
| 无 | 长、结构化、分点、带 markdown |
| 给 5 岁小孩讲 | "机器学习就像教小狗做游戏……"（80 字内） |
| 严谨教授 | "机器学习是人工智能的子集……"（学术、先定义后例子） |

**结论**：System Prompt 能显著改变输出风格，是控制模型行为的主力工具。

### 3.5 软约束 vs 硬约束（本周最重要的认知）

**核心发现**：System Prompt 是**软约束**，不是硬约束。模型会尽量遵守，但复杂/冲突指令下会"打折扣"。

**实测**（`07_task2_priority.py`，要求回答不超过 20 字）：

```
用户: 请写一篇 500 字的文章……
模型: 人工智能始于1956年达特茅斯会议……（约 80 字）   ← 超标！没管住
```

**正确做法：Prompt 引导 + 代码兜底**（`08_task2_hard_constraint.py`）：

| 做法 | 输出 | 字数 | 结果 |
|---|---|---|---|
| 只靠 System Prompt | 人工智能始于1956年达特茅斯会议… | 79 | ❌ 超标 |
| Prompt + 代码截断 | 人工智能（AI）的发展是一部波澜壮阔的科 | **20** | ✅ 达标 |

> 💡 **记住**：Prompt 负责"引导"，代码负责"兜底"。**凡是不能出错的约束，都要在代码里再校验一次。**

### 3.6 结构化输出：三方案选型

LLM 只会输出**文本**，程序需要**可解析的数据**。三种方案：

| 方案 | 保证合法 JSON | 保证字段/类型 | 成本 | 适用 |
|---|:---:|:---:|:---:|---|
| 只靠 Prompt | ❌ | ❌ | 最低 | 不解析、纯展示 |
| **JSON Mode** | ✅ | ❌ | 低 | 简单提取、字段少 |
| **Function Calling** | ✅ | ✅ | 中 | **生产首选** |

**实测翻车**（`10_task3_json_mode.py`）：

```
做法1（只靠 Prompt）原始输出：
```json
{ "姓名": "张三", "年龄": 25, ... }      ← 带 ```json 标记
❌ json.loads 失败: Expecting value: line 1 column 1

做法2（JSON Mode）原始输出：
{"answer":{"姓名":"张三","年龄":"25 岁",...}}   ← 合法但套了 answer 层，且年龄是字符串
✅ 解析成功，但字段类型漂移
```

**Function Calling 精确控制**（`11_task3_function_calling.py`）：

```python
tools = [{
    "type": "function",
    "function": {
        "name": "extract_person_info",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},      # 锁类型
                "skills": {"type": "array", "items": {"type": "string"}},
                "is_student": {"type": "boolean"},
            },
            "required": ["name", "age", "skills", "is_student"],
        },
    },
}]
```

真实输出：`{"name":"张三","age":25,"skills":["Python","JavaScript"],"is_student":true}` —— **age 是整数，四个字段类型全对**。

> 💡 **选型口诀**：能用 Function Calling 就别用 JSON Mode；JSON Mode 适合快速原型；只靠 Prompt 等于赌运气。

### 3.7 LLM 是无状态性的

**关键概念**：每次 API 调用都是**独立的**，模型不记得上一次聊了什么。所谓"多轮对话"，是你**自己把历史消息每次重新塞回去**的。

**实测**（`12_task4_stateless.py`）：

```
>>> 上一轮：用户说"我叫小明"
>>> 这一轮：不传历史，问"我叫什么名字？"
模型: 你还没有告诉我你的名字呀！      ← 失忆
>>> 这一轮：把上一轮消息一起传进去
模型: 你的名字是**小明**。            ← 记得了，因为历史是你塞进去的
```

### 3.8 滑动窗口（控制上下文成本）

对话越长，全塞历史会有三个问题：**撑爆上下文、token 变贵、信息干扰**。

**解法**：只保留最近 N 轮（滑动窗口）。实测 token 对比（`13_task4_sliding_window.py`，20 轮历史）：

| 策略 | 发送消息数 | total_tokens | 回答 |
|---|---|---|---|
| 全留（20 轮） | 41 | **802** | 我们一共聊了 20 轮 |
| 滑动窗口（最近 5 轮） | 11 | **217** | 我们一共聊了 20 轮 |

**省了 73%**。但注意第二次测试里两种都答对了，因为"第 20 轮"恰好就在窗口内 —— 这个测试没真正暴露截断的代价。

**真正暴露代价的测试**（`14_task4_window_tradeoff.py`，问"第 1 轮主题编号是多少"）：

```
全留（20 轮）:        回答: 第 1 轮的主题编号是 1。        total_tokens: 808
滑动窗口（最近 5 轮）: 回答: 第 1 轮的主题编号是 1。        total_tokens: 223
```

> ⚠️ **最危险的发现**：滑动窗口下模型也说"1"，但它**根本不可能知道**（历史里只有 16-20 轮）。它是在**猜**，而且猜得很自信。**截断不会让模型说"我不知道"，它会编一个看起来对的答案** —— 这叫**幻觉（hallucination）**，是 Agent 开发里最阴险的问题之一，W7 会专门治它。

### 3.9 记忆系统

**核心思想**：重要信息（用户偏好、关键结论）**单独存起来**，每次作为 system 或摘要**注入**，而不是依赖原始对话历史。这样即使窗口滑过，重要信息也不丢。

**实现**（`15_task4_manager.py`）：

```python
class ConversationManager:
    def __init__(self, system_prompt, max_turns=5, memory=""):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.memory = memory      # 长期记忆，每次注入
        self.history = []

    def _build_messages(self, user_msg):
        system = self.system_prompt
        if self.memory:
            system += f"\n\n【用户记忆】\n{self.memory}"
        return (
            [{"role": "system", "content": system}]
            + self.history[-(self.max_turns * 2):]   # 滑动窗口
            + [{"role": "user", "content": user_msg}]
        )
```

**实测验证**（窗口设为 3 轮，聊 5 轮）：

```
第 1 轮：我该学什么语言？ → 学 Python（理由：你讨厌 Java）   ← 记忆生效
第 2-4 轮：聊天气/书/笑话（把窗口占满）
第 5 轮：我最讨厌什么语言？ → Java                          ← 窗口已滑过，记忆仍在！
[当前窗口内消息数: 6]   ← 只剩 3 轮
```

> 💡 这就是记忆系统的价值：**不依赖对话历史的"长期记忆"**。W5 会正式展开。

### 3.10 限流与重试

免费/付费 API 都会限流。真实调用中频繁出现：

```
Error code: 429 - {'code': '1305', 'message': '该模型当前访问量过大，请您稍后再试'}
Error code: 429 - {'code': '1113', 'message': '余额不足或无可用资源包'}
```

**正确做法：指数退避重试**：

```python
def call_llm(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(...)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)   # 1s, 2s, 4s 退避
```

> 💡 生产环境**必须**带重试 + 退避。这是 W1 里最容易被忽略、上线后最先炸的东西。

---

## 四、完整代码汇总

### 4.1 通用调用模板（本项目所有代码的基础）

```python
import os, time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "glm-4.7-flash"

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

def call_llm(messages, temperature=0.5, max_tokens=300, extra=None):
    for attempt in range(3):
        try:
            kwargs = dict(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
            if extra:
                kwargs.update(extra)
            return client.chat.completions.create(**kwargs)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
```

### 4.2 综合 API 服务（Task 5 最终产物）

见 `16_task5_api.py`。架构：

```
POST /chat  →  System Prompt + 记忆注入 + 滑动窗口 + 重试 + 硬约束校验  →  返回回复
GET  /health →  健康检查 + 会话数
POST /reset/{session_id}  →  清空会话
```

**端到端真实输出**（验证通过）：

```bash
POST /chat {"session_id":"u1","message":"你好"}
→ {"reply":"您好！我是电商客服小智。请问有什么可以帮您的吗？","history_count":2}

POST /chat {"session_id":"u1","message":"我的订单什么时候发货"}
→ {"reply":"亲，您的订单预计24小时内发货，请耐心等待哦~","history_count":4}

POST /chat {"session_id":"me","message":"我叫阿华，买了一件衣服"}
POST /chat {"session_id":"me","message":"今天天气不错"}
POST /chat {"session_id":"me","message":"我买了什么来着？"}
→ {"reply":"阿华，您买的是一件衣服哦。请问您是想了解物流进度，还是需要退换货呢？"}
```

**第 3 轮是关键证据**：窗口已经滑过，它还记得"一件衣服" —— 记忆系统生效。

---

## 五、学习过程记录（关键实验时间线）

| 阶段 | 做了什么 | 真实发现 |
|---|---|---|
| 1 | 跑 Mock 版 | 验证代码结构，无 API 也能学 |
| 2 | 探测可用模型 | 付费全 429，免费 flash 可用；推理模型空输出 |
| 3 | 第一次真实调用 | `reasoning_tokens=500`，content 为空 |
| 4 | 关思考重跑 | 800 token → 88 token |
| 5 | System Prompt 对比 | 同一问题，角色不同输出天差地别 |
| 6 | 软/硬约束实验 | 79 字管不住，代码截断到 20 字 |
| 7 | 结构化输出 | ```json 翻车 → JSON Mode 类型漂移 → Function Calling 全对 |
| 8 | 无状态实验 | 不传历史就"失忆" |
| 9 | 滑动窗口 | 802→217 token，省 73% |
| 10 | 截断幻觉 | 窗口外的问题也"自信地猜" |
| 11 | 记忆注入 | 窗口滑过，"讨厌 Java"仍答对 |
| 12 | API 端到端 | 多轮记忆验证通过 |

---

## 六、踩坑记录

| 坑 | 现象 | 解决 |
|---|---|---|
| 推理模型空输出 | `content` 空，`reasoning_tokens` 吃掉全部额度 | 关思考，或给足 max_tokens |
| 免费模型限流 | `429 访问量过大` | 指数退避重试 |
| 免费模型余额 | `429 余额不足`（付费模型） | 只用 flash 系列 |
| JSON 翻车 | 输出带 ```` ```json ```` 标记，`json.loads` 崩 | response_format 或 Function Calling |
| 类型漂移 | age 变成 `"25 岁"` 字符串 | Function Calling 锁 schema |
| 截断幻觉 | 窗口外的问题也"自信地猜" | 需专门处理（W7 评测） |
| Key 泄露风险 | 硬编码在代码里 | 一律写 `.env`，已进 `.gitignore` |

---

## 七、核心结论（记住这 5 条）

1. **Prompt 负责引导，代码负责兜底** —— 不能出错的约束必须在代码里校验。
2. **能用 Function Calling 就别用 JSON Mode** —— 后者不保证字段类型。
3. **滑动窗口省钱但会丢信息** —— 重要信息单独存成**记忆**注入。
4. **生产代码必须带重试 + 退避** —— API 一定会限流。
5. **凡是要成本的地方，先问**：这个任务值不值得开"思考"？

> **一句话总结 W1**：你现在能"做出东西"了，但还不能"证明东西对"。**从 W2 开始补这个。**

---

## 八、知识地图（W1 → W2 衔接）

```
W1 已掌握
├── LLM API 调用 + 成本控制
├── Prompt 工程（软/硬约束）
├── 结构化输出
├── 上下文工程（滑动窗口 + 记忆）
└── 工程化意识（重试、限流、API）

W2 要补的最大缺口：知识边界
└── 模型只知道训练数据里的东西
    └── RAG：把外部知识检索出来塞进 Prompt
        文档 → 切片 → Embedding → 向量库 → 检索 → 拼进 Prompt → 生成
```

---

## 附录：week1 文件清单

| 文件 | 用途 |
|---|---|
| `00_check_zhipu.py` | 探测可用模型 |
| `01_hello_glm.py` | Task 1 真实调用（主用） |
| `01_hello_llm.py` | Task 1 DeepSeek 版（参考） |
| `01_hello_llm_mock.py` | 无 Key 时跑通结构 |
| `mock_llm.py` | 统一 Mock 模块 |
| `06_task2_system_prompt.py` | System Prompt 对比 |
| `07_task2_priority.py` | 优先级/防注入 |
| `08_task2_hard_constraint.py` | 软 vs 硬约束 |
| `09_task2_mini_project.py` | 小红书文案生成器 |
| `10_task3_json_mode.py` | JSON Mode |
| `11_task3_function_calling.py` | Function Calling |
| `12_task4_stateless.py` | 无状态性 |
| `13_task4_sliding_window.py` | 滑动窗口 token 对比 |
| `14_task4_window_tradeoff.py` | 截断的代价 |
| `15_task4_manager.py` | 对话管理器（可复用） |
| `16_task5_api.py` | 综合 API 服务 |
| `probe_thinking.py` | 推理模型探测 |
| `requirements.txt` | 依赖 |
| `.env.example` | 环境变量模板 |
