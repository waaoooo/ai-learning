# 提示工程一页纸速查（企业版）

## 1. 三铁律
- 具体 > 聪明
- 说要做什么，不说不做什么
- 分解 > 一次性

## 2. 参数（二选一）
- temperature ↔ top_p
- frequency penalty ↔ presence penalty
- QA 任务：temperature 0.0-0.3

## 3. 四要素
指令 / 上下文 / 输入数据 / 输出指示

## 4. 技术选型
| 需求 | 技术 |
|---|---|
| 简单 | Zero-shot |
| 格式约束 | Few-shot |
| 推理 | CoT |
| 高准确率 | Self-Consistency |
| 实时/私有知识 | RAG |
| 外部操作 | ReAct/Agent |
| 精确计算 | PAL |
| 复杂规划 | ToT/Chaining |

## 5. Agent 三组件
规划 + 工具 + 记忆

## 6. Workflow vs Agent
- 生产优先 Workflow（可控）
- 开放任务用 Agent
- 混合架构

## 7. 上下文工程五原则
消除歧义 / 期望显式化 / 可观测性 / 迭代 / 平衡灵活与约束

## 8. 风险三件套
- 注入/泄漏/越狱 → 红队
- 幻觉 → RAG + 低温 + 引用
- 偏见 → 示例平衡 + 顺序随机

## 9. 企业流程
定义 → 基线 → 迭代 → 评估 → 红队 → 上线 → 优化

## 10. 反模式（别做）
模糊指令 / 一次性复杂prompt / 过度约束 / 盲目CoT / 示例偏斜 / 不评估 / 机密放prompt
