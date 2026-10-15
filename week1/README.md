# Week 1: LLM API 集成 + Prompt Engineering 基础

> 状态：✅ 已完成并验收 · 2026-10-14 · 模型 glm-4.7-flash（智谱，免费）

## 环境

```
Python 3.9.6 · openai 2.x · fastapi · uvicorn · pydantic 2.x
API: https://open.bigmodel.cn/api/paas/v4/   (ZHIPU_API_KEY 填在 .env)
默认模型: glm-4.7-flash，extra_body={"thinking":{"type":"disabled"}}
```

## 任务清单

| 任务 | 主题 | 状态 | 关键产出 |
|------|------|:---:|------|
| Task 1 | 真实 API 调用 + temperature + token | ✅ | `01_hello_glm.py` |
| Task 2 | System Prompt + 软/硬约束 | ✅ | `06/07/08/09_*.py` |
| Task 3 | 结构化输出（JSON Mode / Function Calling） | ✅ | `10/11_*.py` |
| Task 4 | 多轮对话 + 滑动窗口 + 记忆 | ✅ | `12/13/14/15_*.py` |
| Task 5 | 综合 API 服务 | ✅ | `16_task5_api.py` |

## 验收结果

- 真实 API 调用成功，理解 prompt/completion/total token
- temperature 差异：开放创作才明显，简单任务几乎无差
- 关思考：token 从 ~800 降到 ~20（同一任务）
- System Prompt 软约束 vs 代码硬约束，实测 79字 → 20字
- 滑动窗口：20轮 808 token → 5轮 223 token
- 记忆注入：窗口滑过后"讨厌 Java"仍答对
- API 端到端：多轮记忆验证通过

## 踩过的坑（真实）

| 坑 | 现象 | 解决 |
|---|---|---|
| 推理模型空输出 | `content` 为空，`reasoning_tokens=500` | 关思考，或给足 max_tokens |
| 免费模型限流 | `429 访问量过大` | 指数退避重试 |
| 免费模型余额 | `429 余额不足`（glm-4.6 等付费） | 只用 flash 系列 |
| JSON 翻车 | 输出带 ```json 标记，json.loads 崩 | response_format 或 Function Calling |
| 类型漂移 | age 变成 "25 岁" 字符串 | Function Calling 锁 schema |
| 截断幻觉 | 窗口外的问题也"自信地猜" | 需专门处理（W7 评测） |

## 核心结论（记住这几条）

1. Prompt 负责**引导**，代码负责**兜底** —— 不能出错的约束必须在代码里校验
2. 能用 **Function Calling** 就别用 JSON Mode，后者不保证字段类型
3. **滑动窗口**省钱但会丢信息 → 重要信息单独存成**记忆**注入
4. 生产代码**必须带重试 + 退避**（免费/付费 API 都会限流）
5. 凡是要**成本**的地方，先问：这个任务值不值得开"思考"？

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env        # 填入 ZHIPU_API_KEY
python3 01_hello_glm.py     # 跑通第一个真实调用
```

## 下一步

Week 2：RAG 系统（检索增强生成）—— 让模型回答"它本来不知道"的知识。
