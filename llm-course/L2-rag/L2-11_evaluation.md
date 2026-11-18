# L2-11 · RAG 应用评估（Ragas / Trulens）

> **所属阶段**：L2 RAG 开发实战工坊
> **本课目标**：建立完整的 RAG 评估体系——从指标定义、公式、实现到工具（Ragas/Trulens）与行业阈值，并能针对真实系统做误差分析与优化。
> **先修**：L2-01 ~ L2-10；建议先读 L2-07（痛点诊断）。
> **预计时长**：90-120 分钟（含实验）

---

## 1. 为什么评估是 RAG 的"下半场"

很多团队把精力放在"把 RAG 跑起来"，但上线后才发现：答案时好时坏、无法量化改进、不知道该调 chunk 还是调 prompt。**没有评估，优化就是碰运气。**

评估要回答三个问题：
1. **检索对不对**（召回的上下文是否相关、是否完整）
2. **生成对不对**（答案是否忠实、是否切题）
3. **改动有没有变好**（回归验证）

---

## 2. 评估方法

### 2.1 人工评估
- **做法**：标注员对答案打分（相关性/忠实度/完整性），或 A/B 对比
- **优点**：质量天花板高、能发现细微问题
- **缺点**：慢、贵、主观、不可回归
- **适用**：冷启动建集、争议样本仲裁、线上抽检

### 2.2 自动化评估
- **做法**：用指标或 LLM-as-judge 批量计算
- **优点**：快、可回归、可 CI
- **缺点**：指标与真实质量有偏差、需要评测集
- **适用**：迭代回归、版本对比

### 2.3 LLM-as-judge
- **做法**：用强模型（如 GPT-4/Claude）对答案打分或两两比较
- **注意**：位置偏差、长度偏差、自我偏好；需固定 judge 版本、随机化顺序、多 judge 投票

### 2.4 混合策略（推荐）
```
冷启动：人工标注 50-200 条评测集
迭代期：自动化指标回归 + LLM-judge 抽审
上线后：真实反馈回流，持续补充评测集
```

---

## 3. 评估集构建

### 3.1 数据来源
- **真实 query 采样**：从日志中采样高频/边界 query（最佳）
- **合成 query**：LLM 基于文档生成问题 + 人工校验
- **边界样本**：无答案问题、多跳问题、歧义问题

### 3.2 标注字段
```json
{
  "question": "年假怎么算？",
  "ground_truth": "入职满1年5天，满3年10天，满5年15天。",
  "relevant_docs": ["年假制度"],
  "answer_type": "fact"
}
```

### 3.3 规模建议
- 冷启动 50-100 条；迭代期 200-500 条；覆盖 6-8 类问题
- 每条标注：标准答案 + 相关文档 + 问题类型

---

## 4. 检索评估指标（精确定义）

### 4.1 上下文相关性 Context Relevancy
- **定义**：检索到的 chunk 中，与问题相关的比例
- **公式**：`相关 chunk 数 / 检索 chunk 总数`
- **关注**：检索噪声
- **阈值经验**：>0.7 较好；<0.5 说明检索质量差

### 4.2 上下文精度 Context Precision
- **定义**：相关 chunk 在 top-k 中的排序质量（越靠前越好）
- **公式**（类 MAP）：对每个相关位置 i，计算 `precision@i`，取平均
- **关注**：top 位置是否精准
- **经验**：>0.8 较好

### 4.3 上下文召回率 Context Recall
- **定义**：标准答案所需的信息被检索到的比例
- **公式**：`被检索到的关键信息 / 标准答案所需关键信息`
- **关注**：是否漏召回
- **经验**：>0.8 较好

### 4.4 检索指标的关系
```
Relevancy 高 + Precision 高 → 检索又准又不噪
Recall 高 → 该找的都找到了
三者常此消彼长（k 越大 Recall↑ 但 Relevancy↓）
```

---

## 5. 响应评估指标（精确定义）

### 5.1 忠实度 Faithfulness
- **定义**：答案中的每个陈述能否在检索上下文中找到依据
- **做法**：把答案拆成若干陈述 → 每条判断是否被上下文支持 → 支持比例
- **关注**：幻觉
- **经验**：>0.9 较好；<0.7 严重幻觉

### 5.2 答案相关性 Answer Relevancy
- **定义**：答案是否切题、完整回应问题
- **做法**：LLM 从答案反推可能的问题，与原始问题做 embedding 相似度；或直接 judge
- **关注**：答非所问、漏答
- **经验**：>0.85 较好

### 5.3 其他响应指标
- **答案正确性 Answer Correctness**：与 ground truth 对比（语义相似度或 LLM 判分）
- **完整性 Completeness**：是否覆盖 ground truth 的所有要点
- **简洁性 Conciseness**：是否冗余

---

## 6. 指标总结与行业阈值（经验值，待按场景校准）

| 指标 | 层级 | 优秀 | 及格 | 说明 |
|---|---|---|---|---|
| Context Relevancy | 检索 | ≥0.85 | ≥0.7 | 噪声控制 |
| Context Precision | 检索 | ≥0.8 | ≥0.6 | top 排序 |
| Context Recall | 检索 | ≥0.9 | ≥0.75 | 召回完整 |
| Faithfulness | 响应 | ≥0.95 | ≥0.85 | 防幻觉 |
| Answer Relevancy | 响应 | ≥0.9 | ≥0.8 | 切题 |

> ⚠️ 阈值因场景而异：医疗/法律更严（Faithfulness ≥0.95），闲聊可放宽。**先跑基线再定阈值。**

---

## 7. 评估工具

### 7.1 Ragas（推荐入门）
四个核心指标：`faithfulness`、`answer_relevancy`、`context_precision`、`context_recall`。

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

data = {
    "question": ["年假怎么算？"],
    "answer": ["入职满1年5天，满3年10天，满5年15天。"],
    "contexts": [["公司年假：入职满1年享有5天年假，满3年10天，满5年15天。"]],
    "ground_truth": ["年假按入职年限计算。"],
}
result = evaluate(Dataset.from_dict(data),
                  metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(result)
```

**Ragas 原理**：用 LLM 做 judge（faithfulness 拆句验证、answer_relevancy 反向生成问题算相似度），因此需要可用 LLM。

### 7.2 Trulens
- 从"检索-生成"全链路打分，支持自定义 feedback function
- 指标：`groundedness`（忠实度）、`context_relevance`、`answer_relevance`
- 可接入监控看板，适合生产持续观测

> Ragas 偏离线评测/回归；Trulens 偏在线观测。两者可互补。

---

## 8. 应用实战一：医疗 RAG 评估

### 8.1 场景
医疗问答对幻觉零容忍，需重点看 Faithfulness 与 Context Recall。

### 8.2 诊断流程
1. 建 100 条评测集（含 20% 无答案问题）
2. 跑 Ragas 四指标，得基线
3. **按指标定位瓶颈**：
   - Context Recall 低 → 检索问题（换混合检索/调 chunk）
   - Faithfulness 低 → 生成问题（prompt 约束/上下文质量）
   - Answer Relevancy 低 → prompt 或检索噪声
4. 针对性优化后回归，确认指标不退化

### 8.3 优化策略
- 分阶段：先修检索（Pre/Post-Retrieval），再修生成（prompt/模型）
- 核心原则：**每次只改一个变量**，用评测集验证

---

## 9. 应用实战二：知识块大小评估

### 9.1 为什么重要
chunk_size 同时影响检索精度与上下文完整性，是最值得调的超参。

### 9.2 评估方法
固定其他变量，对 chunk_size ∈ {200, 500, 800, 1200} 分别：
1. 建索引
2. 跑同一评测集
3. 记录 Context Precision / Recall / Faithfulness
4. 画曲线，选最优点

### 9.3 经验
- 小 chunk：Precision 高、Recall 低
- 大 chunk：Recall 高、Precision 低、token 成本高
- 常用 500-1000 字符 + 10-20% overlap

---

## 10. 应用实战三：2023 全球智能汽车 AI 挑战赛

### 10.1 背景
赛题涉及知识问答，队伍用 RAG。关键提升点：**重排序模型**。

### 10.2 结果分析
- 初版（纯向量）：召回够但 top 不准，答案相关性低
- 加入 reranker（cross-encoder）后：top-3 相关性显著提升，最终指标提升

### 10.3 结论
- 检索"召回"与"排序"是两件事；重排是低成本高收益的优化。
- 竞赛/生产都适用：**粗召回 + 精排序**。

---

## 11. 最小可运行示例

配套脚本：`L2-11_evaluation.py`。运行：

```bash
cd /Users/freedom/hermes/ai-learning/llm-course/L2-rag
python L2-11_evaluation.py
```

脚本**手工实现** 5 个指标（无 ragas 依赖），对样本计算分数，并附 Ragas 等价代码注释。

---

## 12. 实验与练习

1. 为 L2-05 的 HR 系统建 10 条评测集，计算 5 指标，找最弱项。
2. 固定其他变量，把 chunk_size 从 200 调到 1000，记录指标变化。
3. 加入"无答案"样本，验证 Faithfulness 是否被正确评估。
4. 用 LLM-as-judge 对同一批答案打分，与自动化指标对比，观察偏差。

---

## 13. 小结

- 评估分检索（Relevancy/Precision/Recall）与响应（Faithfulness/Answer Relevancy）两层。
- 人工建集 + 自动化回归 + LLM-judge 抽审是推荐组合。
- Ragas 适合离线回归，Trulens 适合在线观测。
- 先跑基线，再按指标定位瓶颈，每次只改一个变量。

---

## 14. 自测题

1. Context Recall 与 Answer Relevancy 的区别？
2. Faithfulness 为什么比 Answer Relevancy 更难评估？
3. 自动化评估能否完全替代人工？为什么？
4. 如何构建一个 RAG 评测集？（数据来源/标注方法）
5. 如果 Context Recall 低但 Faithfulness 高，说明什么？该优化哪一环？

### 自测题答案

1. **区别**：Context Recall 衡量"标准答案所需信息是否被检索到"（检索层）；Answer Relevancy 衡量"生成答案是否切题完整"（响应层）。
2. **Faithfulness 更难**：需逐句判断答案陈述是否有上下文依据，涉及语义推理，主观性强，常需 LLM-as-judge。
3. **不能完全替代**：自动化指标与真实质量有偏差，无法覆盖表达/语气/合规；应自动化做回归、人工做抽审。
4. **构建方法**：真实 query 采样 + LLM 合成 + 边界样本；标注 question/ground_truth/relevant_docs/type；冷启动 50-200 条。
5. **说明**：检索漏了信息，但模型没有编造（只答了找到的部分）；应优化**检索**（混合检索、调 chunk、查询扩展），而非生成。

---

## 15. 延伸阅读

- Ragas docs: https://docs.ragas.io
- Trulens docs: https://www.trulens.org
- "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (Es et al., 2023)

---

## 16. 验收动作

- [ ] 跑通 `L2-11_evaluation.py`，理解 5 个指标的计算
- [ ] 为 L2-05 HR 系统建 10 条评测集并计算指标
- [ ] 完成实验 2（chunk_size 扫描），画出指标曲线
- [ ] 尝试安装 ragas 并跑一次（可选，注意 Python 版本）
