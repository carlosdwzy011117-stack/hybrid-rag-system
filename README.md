# Hybrid RAG Retrieval System

**Work in progress — Day 7 / 21**

从零构建支持稀疏 + 稠密混合检索的端到端 RAG 系统，不依赖 LangChain / LlamaIndex 等高层框架。

## 当前进度

| 阶段 | 方法 | Recall@10 | MRR | NDCG@10 |
|------|------|-----------|-----|---------|
| Day 3 | BM25 (sparse) | 0.6862 | 0.5242 | 0.5597 |
| Day 6 | Dense (BGE-small-en-v1.5) + FAISS IndexFlatIP | 0.8452 | 0.6845 | 0.7200 |
| Day 7 | Hybrid (RRF, k=60, pool=100) | 0.7877 | 0.6283 | 0.6617 |
| **Day 7** | **Weighted (min-max + α=0.2)** | **0.8502** | **0.6934** | **0.7269** |
| Day 7 | Weighted (α=0.3, Recall optimum) | 0.8562 | 0.6850 | 0.7239 |
| Day 14 (planned) | + Cross-encoder reranker | — | — | — |

Dataset: BEIR / SciFact — 5,183 docs / 300 queries (test split)

## Day 7 实验亮点：从 RRF 到 Weighted Sum

实验过程发现 RRF 在 SciFact 上**反而退化** —— Recall@10 比 Dense 单跑下降 6.8%。深入分析定位根因：SciFact 上 BM25 召回质量极弱（部分 gold doc 在 BM25 top-500 之外），RRF 的"信任两路共识"特性反而稀释了 Dense 的强信号。

解决方案：将融合方式从 rank-based RRF 改为 score-based Weighted Sum + min-max 归一化，并扫描 α 找最优值。

### α 扫描结果（同一份 BM25/Dense 索引复用，避免重建）

| α | Recall@10 | MRR | NDCG@10 | 说明 |
|---|-----------|-----|---------|------|
| 0.0 | 0.8452 | 0.6845 | 0.7200 | = Dense baseline (sanity check ✓) |
| 0.1 | 0.8486 | 0.6913 | 0.7252 | |
| **0.2** | **0.8502** | **0.6934** | **0.7269** | **MRR & NDCG 最优** |
| **0.3** | **0.8562** | 0.6850 | 0.7239 | **Recall 最优** |
| 0.5 | 0.8246 | 0.6590 | 0.6952 | 性能开始下降 |
| 0.7 | 0.7585 | 0.6019 | 0.6352 | |
| 1.0 | 0.6862 | 0.5242 | 0.5597 | = BM25 baseline (sanity check ✓) |

**关键观察**：α=0.0 / 1.0 两端点结果与单路 baseline **完全一致到第 4 位小数**，提供 WeightedRetriever 算法正确性的隐式验证。最优 α=0.2 在所有三个指标上同时超过 Dense baseline 与 RRF Hybrid。

## 项目动机

针对 AI Agent 在外部知识库检索中的核心矛盾：
- **稀疏检索 (BM25)** 对专有名词、ID、数字匹配精准，但**语义盲**（同义词、改写无法识别）
- **稠密检索 (Dense)** 善于语义匹配，但**对罕见词、连字符切分、专有名词较弱**

本项目从零实现混合检索方案，**所有检索器、融合算法、评估指标均手写**，目的是吃透 RAG 每一层的原理与工程细节，而非作为框架调用方。

**Day 3 真实观察**：BM25 在 query `"0-dimensional biomaterials show inductive properties"` 上未能召回 gold doc — 是后续引入 Dense 检索的核心动机。

## 项目结构

```
hybrid-rag-system/
├── data/scifact/                          # BEIR dataset (gitignored)
├── src/
│   ├── data_loader.py                     # BEIR 数据加载
│   ├── evaluator.py                       # Recall@K / MRR / NDCG@K
│   └── retrievers/
│       ├── bm25_retriever.py              # BM25 (sklearn 风格 API)
│       ├── dense_retriever.py             # Dense + FAISS IndexFlatIP
│       ├── hybrid_retriever.py            # RRF 融合
│       └── weighted_retriever.py          # min-max + 加权和
├── tests/test_evaluator.py                # 17 个 pytest 单元测试
└── scripts/
    ├── run_bm25_baseline.py               # 端到端 baseline
    ├── run_dense_baseline.py
    ├── run_hybrid_baseline.py
    ├── run_weighted_baseline.py
    ├── smoke_test_dense.py                # 单 query 验证（已知答案）
    ├── smoke_test_hybrid.py
    └── sweep_weighted_alpha.py            # α 超参扫描
```

## 技术栈

Python 3.10 · BM25 (rank_bm25) · Sentence-Transformers (BGE-small-en-v1.5) · FAISS (IndexFlatIP) · GPT-4o-mini · BEIR · RAGAS · pytest

## 快速开始

环境配置：
```bash
conda create -n rag python=3.10
conda activate rag
pip install rank_bm25 beir sentence-transformers faiss-cpu numpy pytest
```

跑各路 baseline：
```bash
python scripts/run_bm25_baseline.py
python scripts/run_dense_baseline.py
python scripts/run_hybrid_baseline.py        # RRF 融合
python scripts/run_weighted_baseline.py      # Weighted 融合 (α=0.3)
python scripts/sweep_weighted_alpha.py       # α 超参扫描
```

跑评估指标单元测试：
```bash
pytest tests/ -v
```

## 开发路线图 (21 Days)

- [x] **Day 1**: 项目结构、Git 仓库、依赖管理
- [x] **Day 2**: 评估框架 (Recall@K / MRR / NDCG@K) + 17 个 pytest 单元测试
- [x] **Day 3**: BEIR SciFact 数据加载 + BM25 检索器 (sklearn 风格 API) + 端到端 baseline
- [x] **Day 4-5**: Sentence-Transformers + BGE-small-en-v1.5 探索（query 1 详细分析）
- [x] **Day 6**: DenseRetriever 类 + FAISS IndexFlatIP + Dense baseline
- [x] **Day 7**: HybridRetriever (RRF) + WeightedRetriever (min-max + 加权和) + α 扫描 + 算法对比分析
- [ ] **Day 8-10**: 评估升级、RAGAS、错误案例分析
- [ ] **Day 11-14**: Cross-encoder 重排序 + 消融实验
- [ ] **Day 15-18**: 端到端 RAG (GPT-4o-mini) + RAGAS 评估生成质量
- [ ] **Day 19-21**: 错误案例分析、文档完善、对外发布

## 设计决策记录

**为什么不用 LangChain / LlamaIndex？**
框架封装会跳过 BM25 分词、IDF 计算、归一化、融合算法等关键细节。本项目从底层手写每一层，目标是建立对 RAG 系统的完整心智模型，而非熟练调用 API。

**为什么 BM25 索引时拼接 title + text？**
BEIR / SciFact 的 title 是高密度信号（每词都关键），仅对 text 建索引会丢失 title 关键词。这是 BEIR 论文官方 baseline 的标准做法。

**为什么 Dense 用 BGE-small-en-v1.5 + FAISS IndexFlatIP？**
- BGE 输出已 L2 归一化，IndexFlatIP（内积）等价于余弦相似度，避免重复归一化的浮点损失。
- 用 IndexFlatIP 而非 IndexFlatL2 是因为 BGE 训练目标是余弦相似度，L2 距离的几何含义会失真。

**为什么 RRF 在 SciFact 上反而退化（Day 7 实验结论）？**
RRF 隐含假设两路 retriever 实力相当。但 SciFact 上 BM25 系统性弱（部分 gold doc 在 BM25 top-500 之外），RRF 的"两路 rank 都靠前才得高分"反而让 Dense 的强信号被两路都中等的"BM25 假阳性"压过。这是有价值的负向实验结论。

**为什么 Weighted Sum 比 RRF 在 SciFact 上更优？**
RRF 用 rank（等距量化）丢失了原始分数的"断崖"信息。例如 Dense 余弦 0.7 vs 0.5 的差异在 RRF 中等价于"rank 差几名"的微弱信号，但 min-max 归一化后差异 = 0.4 vs 0.0，显式保留。α 旋钮让弱路（BM25）权重可调，避免 RRF 的隐式 50/50。两者配合是 Day 7 反超 Dense 的关键。

**为什么 α 扫描复用同一份 BM25/Dense 索引？**
WeightedRetriever 通过 composition（持有子 retriever 实例）设计，扫描 α 时只重新构造 Weighted 这一层壳、不重建索引。7 个 α 的扫描时间从 ~15 分钟（独立 baseline）降到 ~5 分钟。

## License

MIT
