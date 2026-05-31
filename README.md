# Hybrid RAG System on BEIR/SciFact

端到端检索增强生成（RAG）系统，包含稀疏 + 稠密混合检索、α 加权融合调优、LLM 答案生成，以及对检索和生成质量的量化评估 —— 所有核心组件均手写实现，未依赖 LangChain / LlamaIndex 等高层框架。

<!-- TODO Day 19: 插入 demo GIF 或截图 -->
<!-- ![demo](docs/demo.gif) -->

## 项目亮点

- **四种检索方法系统对比**：BM25 → Dense (BGE-small) → Hybrid (RRF) → Weighted Fusion (α 调优)，最优方案在 NDCG@10 上**比 BM25 baseline 提升 29%**。
- **端到端 RAG 链路**：检索 → 上下文构造 → LLM 生成（DeepSeek / GPT-4o-mini）→ RAGAS 评估 faithfulness 与 answer relevancy。
- **负向结果深度分析**：RRF 在 SciFact 上反而退化 —— 定位根因为 BM25 在该数据集上的系统性弱召回，并通过切换到 score-based Weighted Sum + α 调优解决。
- **LLM-as-judge 稳定性 pilot study**：5 query × 5 重复的方差分解实验，发现 faithfulness 的 between/within 方差比约为 9 倍，由此提出 RAG 评估需多次取平均的工程结论。
- **不用框架封装**：LangChain/LlamaIndex 仅作为 RAGAS 内部依赖；所有检索器、融合算法、评估指标、RAG 链路均手写实现。

---

## 检索效果对比

数据集：**BEIR/SciFact** —— corpus 5,183 篇，test set 300 query。

| 方法 | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|
| BM25（rank_bm25） | 0.6862 | 0.5242 | 0.5597 |
| Dense（BGE-small-en-v1.5 + FAISS IndexFlatIP） | 0.8452 | 0.6845 | 0.7200 |
| Hybrid RRF（k=60, pool=100） | 0.7877 | 0.6283 | 0.6617 |
| **Weighted Fusion（min-max + α=0.2）** | **0.8502** | **0.6934** | **0.7269** |
| Weighted Fusion（α=0.3, Recall 最优） | 0.8562 | 0.6850 | 0.7239 |

**核心结果**：Weighted Fusion 在 α=0.2 时同时取得 MRR 和 NDCG@10 最优，相比 BM25 baseline 提升 29.9% NDCG@10，相比 Dense baseline 提升 1.0%。α=0.3 时 Recall@10 达到峰值 0.8562。

### α 扫描：从负向结果到正向解法

最初采用 RRF 融合，Recall@10 **反而下降** 6.8%（vs Dense 单跑）—— 这是个值得深挖的负向结果，而非可以忽略的失败实验。根因分析：

> SciFact 的 query 对 BM25 不友好（短科学声明、连字符术语、改述型 gold doc），不少 gold doc 落在 BM25 top-500 之外。RRF "信任两路共识"的假设在一路系统性弱时就失效 —— Dense 的强信号被 BM25 的噪声稀释。

解法：从 rank-based 的 RRF 切换为 **score-based Weighted Sum + min-max 归一化 + α 调优**。

<!-- TODO Day 20: 插入 alpha_sweep.png 图 -->

| α | Recall@10 | MRR | NDCG@10 | 备注 |
|---|---|---|---|---|
| 0.0 | 0.8452 | 0.6845 | 0.7200 | = Dense baseline（sanity check ✓） |
| 0.1 | 0.8486 | 0.6913 | 0.7252 | |
| **0.2** | **0.8502** | **0.6934** | **0.7269** | MRR & NDCG 双最优 |
| **0.3** | **0.8562** | 0.6850 | 0.7239 | Recall 最优 |
| 0.5 | 0.8246 | 0.6590 | 0.6952 | 开始下降 |
| 0.7 | 0.7585 | 0.6019 | 0.6352 | |
| 1.0 | 0.6862 | 0.5242 | 0.5597 | = BM25 baseline（sanity check ✓） |

**两端 sanity check**：α=0.0 与 α=1.0 的结果分别与对应单路 baseline 精确匹配到第 4 位小数，对融合算法实现正确性提供隐式验证。

---

## 端到端 RAG 链路

```
   Query
     ↓
[ BM25 / Dense / Hybrid / Weighted ]
     ↓
  Top-K Documents (默认 K=5)
     ↓
[ Prompt 构造："基于以下文档作答，未提及则直接说不知道" ]
     ↓
[ LLM Generator（DeepSeek-chat / GPT-4o-mini，OpenAI-compatible API） ]
     ↓
  Final Answer
     ↓
[ RAGAS 评估：faithfulness + answer relevancy ]
```

Generator 通过 `(base_url, api_key, model)` 三件套抽象实现 provider 无关 —— DeepSeek 与 OpenAI 之间切换只改配置，不改代码。API key 通过环境变量 `OPENAI_API_KEY` 管理，不落盘、不进项目目录。

### 生成质量评估（RAGAS，示例 query）

对一条代表性 query，由 DeepSeek 作为裁判 LLM、BGE-small 作为 embedding 模型，评估结果：

| 指标 | 取值 | 解读 |
|---|---|---|
| Faithfulness | 0.67 – 1.00（跨 run 波动） | 答案中的 claim 是否被检索到的 context 支持 |
| Answer Relevancy | 0.00 | 答案为"文档中无相关信息"，被识别为 non-committal |

**这个 case 暴露出的关键洞察**：faithfulness=1.0 同时 answer_relevancy=0.0 并非矛盾 —— 答案完全忠实于 context（无幻觉），但没有回答问题。这正好印证了两个指标作为互补信号的必要性。

---

## Pilot Study：LLM-as-Judge 稳定性

**动机**：RAGAS 用 LLM 作为裁判。单次 RAGAS 评分到底可信吗？为了用数据回答这个问题，设计了一个方差分解实验：

- **实验设计**：固定 `(query, answer, retrieved_docs)` 三元组，只重跑 RAGAS 评估本身 —— 这样可以隔离掉检索/生成的方差，单独测量裁判 LLM 自身的不稳定性。规模 **5 query × 5 重复 = 25 次评估**，然后拆分 between-query 方差与 within-query 方差。

- **实验结果**（按 query 列出 mean / std）：

| QID | 类型 | Faithfulness (mean / std) | Answer Relevancy (mean / std) |
|---|---|---|---|
| 49 | SUPPORT | 0.97 / 0.064 | 0.897 / 0.0006 |
| 124 | SUPPORT | 1.00 / 0.000 | 0.614 / 0.004 |
| 42 | CONTRADICT | 0.63 / 0.075 | 0.000 / 0.000 |
| 57 | CONTRADICT | 0.77 / 0.072 | 0.836 / 0.0001 |
| 137 | CONTRADICT | 1.00 / 0.000 | 0.548 / 0.000 |

- **结论**：
  - **Faithfulness 不稳定**：between-query 方差约为 within-query 方差的 9 倍 —— 裁判自身贡献了不可忽略的随机性。根因：RAGAS 的 claim 拆解步骤非确定性，同一个答案在不同 run 中可能被拆成 1 条 vs 3 条 claim。
  - **Answer Relevancy 稳定**：within-query std 基本为 0，包括对 non-committal 答案的 0.0 硬触发（RAGAS 内置识别机制）。
  - **工程结论**：报告 RAG 生成质量时，faithfulness 必须 N≥3 次取平均；answer relevancy 单次评估即可信。

- **局限性**：pilot 规模（n=5 query、单一裁判模型、固定温度）。下一步：扩到 ≥30 query、引入第二个裁判模型做双因素 ANOVA、并给出效应量的置信区间。

---

## 项目动机

RAG 系统结合检索与生成，但两个组件各自有独立的失败模式：

- **稀疏检索（BM25）** 擅长精确词项匹配（ID、专有名词、数字），但**语义盲** —— 同义词、改述、连字符切分都会漏召回。
- **稠密检索（BGE）** 善于捕捉语义相似，但**对罕见词、OOV 短语较弱**。
- **生成模型** 即使在完美检索下也可能产生幻觉，或者忠实于 context 却答非所问。

本项目**从底层手写**实现混合检索，目的是吃透每一层的机制 —— 分词、IDF、score 归一化、rank 融合 —— 而非作为框架的调用方。每个负向发现（RRF 退化、裁判方差）都成为一次深入调查的契机，而不是绕过去的障碍。

**Day 3 的具体例子**：BM25 在 query `"0-dimensional biomaterials show inductive properties"` 上未召回 gold doc（连字符 + 改述）；BGE-small 把它排到 #6 / 5183（前 0.12%），similarity = 0.7153。这个单一 case 就是后续引入 hybrid 检索的实证动机。

---

## 项目结构

```
hybrid-rag-system/
├── data/scifact/                          # BEIR 数据集（gitignored）
├── src/
│   ├── data_loader.py                     # BEIR 数据加载（corpus / queries / qrels）
│   ├── evaluator.py                       # Recall@K, MRR, NDCG@K, evaluate_retriever()
│   ├── generator.py                       # OpenAI-compatible LLM client
│   └── retrievers/
│       ├── bm25_retriever.py              # BM25, sklearn 风格 API
│       ├── dense_retriever.py             # BGE + FAISS IndexFlatIP
│       ├── hybrid_retriever.py            # RRF 融合
│       └── weighted_retriever.py          # min-max + 加权和
├── tests/
│   ├── test_evaluator.py                  # 17 个 metric 单元测试
│   └── test_evaluate_retriever.py         # 18 个 aggregator 单元测试
├── scripts/
│   ├── run_bm25_baseline.py
│   ├── run_dense_baseline.py
│   ├── run_hybrid_baseline.py
│   ├── run_weighted_baseline.py
│   ├── sweep_weighted_alpha.py            # α 扫描，共享索引避免重建
│   ├── run_rag_pipeline.py                # 端到端 RAG，可作为函数 import
│   └── eval_rag_pipeline.py               # RAGAS 评估，含多次平均
└── app.py                                  # Streamlit demo（TODO Day 19）
```

---

## 关键工程决策

**Retriever 统一 sklearn 风格 API**。四种 retriever 都暴露 `__init__` / `index(corpus)` / `search(query, top_k)`，返回 `[(doc_id, score), ...]`。结果是 `evaluate_retriever()`、α 扫描、下游 pipeline 可以把它们当成可互换的实例处理，无需 `isinstance` 判断。

**融合 retriever 用 composition 而非继承**。`HybridRetriever` 和 `WeightedRetriever` 持有 `BM25Retriever` + `DenseRetriever` 实例，而非继承自它们。直接收益：α 扫描时 7 个值复用同一份 BM25 索引 + 同一份 Dense 索引，扫描时间从 ~15 分钟降到 ~5 分钟。

**Dense 用 BGE + FAISS IndexFlatIP，不用 IndexFlatL2**。BGE 输出已经过 L2 归一化，IndexFlatIP（内积）在单位向量上等价于余弦相似度，省掉一次归一化、并避免浮点漂移。换 IndexFlatL2 反而会引入 L2 距离与余弦的几何错位。

**`evaluate_retriever()` 作为唯一评估入口**。重构前每个 baseline 脚本各自算指标，重构后统一为一个函数，返回 `recall@{1,5,10,20}` / `mrr` / `ndcg@20` 的 dict。重构过程采用 shadow run 验证：新旧两条评估路径并行跑，直到所有 retriever 的数字精确匹配到第 4 位小数，再退役旧路径。4 个 retriever（BM25 / Dense / RRF / Weighted）+ α 扫描端点全部验证零回归。

**LLM provider 抽象**。Generator 配置为 `(base_url, api_key, model)` 三件套，DeepSeek-chat（开发期低成本）和 GPT-4o-mini（面试官熟悉的模型）之间切换只改 3 行配置。API key 全部走环境变量。

**`run_pipeline()` 重构为可 import 的函数，而非裸脚本**。原因：`eval_rag_pipeline.py` 需要 `(query, answer, docs)` 三元组，但不应每次迭代评估逻辑就重跑慢的生成步骤。import 模块会执行其顶层代码，函数则要等到被调用才执行 —— 这是"评估逻辑与链路逻辑分离"的真正工程理由。

---

## 技术栈

Python 3.10 · `rank_bm25` · `sentence-transformers`（BGE-small-en-v1.5）· `faiss-cpu`（IndexFlatIP）· OpenAI-compatible LLM API（DeepSeek / GPT-4o-mini）· `ragas` 0.4.3 · `langchain-huggingface` · BEIR · `pytest` · Streamlit

---

## 快速开始

```bash
# 1. 环境
conda create -n rag python=3.10
conda activate rag
pip install -r requirements.txt

# 2. 数据
# 下载 BEIR/SciFact 并解压到 data/scifact/
# 期望结构：data/scifact/{corpus.jsonl, queries.jsonl, qrels/test.tsv}

# 3. API key（用于生成 + RAGAS 裁判）
# Windows PowerShell:
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_BASE_URL="https://api.deepseek.com"   # 留空则走 OpenAI 官方
# Linux/Mac:
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.deepseek.com

# 4. 跑各路 baseline
python scripts/run_bm25_baseline.py
python scripts/run_dense_baseline.py
python scripts/run_weighted_baseline.py      # α=0.3
python scripts/sweep_weighted_alpha.py       # α ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}

# 5. 端到端 RAG + RAGAS 评估
python scripts/run_rag_pipeline.py
python scripts/eval_rag_pipeline.py          # 已内置多次平均

# 6. Streamlit demo
streamlit run app.py

# 7. 单元测试
pytest tests/ -v                             # 共 35 个测试
```

---

## 项目进度

**Phase 1 — 检索基础设施（✅ 完成）**
- BEIR/SciFact 数据加载，含完整 docstring
- 4 种检索方法在统一 sklearn 风格 API 下实现
- 7 点 α 扫描，复用索引避免重建
- 17 个 metric 单元测试

**Phase 2 — 评估框架（✅ 完成）**
- `evaluate_retriever()` 聚合器，shadow run 验证 4 个 retriever 全部零回归
- 18 个 aggregator 单元测试
- 所有 baseline 脚本统一接入单一评估入口

**Phase 3 — 生成与端到端（✅ 完成）**
- Generator 类含 provider 抽象（DeepSeek / GPT-4o-mini 可切换）
- `run_pipeline()` 返回三元组供下游评估复用，避免重跑慢生成
- RAGAS 集成 faithfulness + answer relevancy
- Embedding 适配器解决 LangChain ↔ RAGAS 接口裂缝（`LangchainEmbeddingsWrapper`）

**Phase 4 — 评估方法学 Pilot（✅ 完成）**
- 方差分解实验设计：固定三元组隔离评估器噪声
- 5 query × 5 重复实验，between/within 方差分解
- 工程结论由数据推导：faithfulness 必须多次取平均

**Phase 5 — 产品化（🚧 进行中）**
- Streamlit demo 支持 retriever 切换
- Demo 录制与文档完善

---

## License

MIT
