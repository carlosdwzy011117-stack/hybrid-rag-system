# Hybrid RAG Retrieval System

> **Work in progress** — Day 3 / 21
> 从零构建支持稀疏 + 稠密混合检索的端到端 RAG 系统，不依赖 LangChain / LlamaIndex 等高层框架。

---

## 当前进度

| 阶段 | 方法 | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|---|
| **Baseline (Day 3)** | BM25 (sparse) | **0.6862** | 0.5242 | 0.5597 |
| Day 7 (planned) | Dense (BGE / MPNet) + FAISS | — | — | — |
| Day 10 (planned) | Hybrid (RRF fusion) | — | — | — |
| Day 14 (planned) | + Cross-encoder reranker | — | — | — |

**Dataset**: [BEIR](https://github.com/beir-cellar/beir) / SciFact — 5,183 docs / 300 queries (test split)

---

## 项目动机

针对 AI Agent 在外部知识库检索中的核心矛盾：

- **稀疏检索 (BM25)** 对专有名词、ID、数字匹配精准，但**语义盲**（同义词、改写无法识别）
- **稠密检索 (Dense)** 善于语义匹配，但**对罕见词、连字符切分、专有名词较弱**

本项目从零实现混合检索方案，**所有检索器、融合算法、评估指标均手写**，目的是吃透 RAG 每一层的原理与工程细节，而非作为框架调用方。

**Day 3 真实观察**：BM25 在 query `"0-dimensional biomaterials show inductive properties"` 上未能召回 gold doc — 是后续引入 Dense 检索的核心动机。

---

## 项目结构

    hybrid-rag-system/
    ├── data/scifact/                       # BEIR dataset (gitignored)
    ├── src/
    │   ├── data_loader.py                  # BEIR 数据加载
    │   ├── evaluator.py                    # Recall@K / MRR / NDCG@K
    │   └── retrievers/bm25_retriever.py    # BM25 检索器 (sklearn 风格 API)
    ├── tests/test_evaluator.py             # 17 个 pytest 单元测试
    └── scripts/run_bm25_baseline.py        # 端到端 baseline

---

## 技术栈

Python 3.10 · BM25 (rank_bm25) · Sentence-Transformers (BGE / MPNet) · FAISS · GPT-4o-mini · BEIR · RAGAS · pytest

---

## 快速开始

环境配置：

    conda create -n rag python=3.10
    conda activate rag
    pip install rank_bm25 beir numpy pytest

跑 BM25 Baseline：

    python scripts/run_bm25_baseline.py

预期输出：

    BM25 Baseline on SciFact (n=300 queries):
      Recall@10: 0.6862
      MRR:        0.5242
      NDCG@10: 0.5597

跑评估指标单元测试：

    pytest tests/ -v

---

## 开发路线图 (21 Days)

- [x] **Day 1**: 项目结构、Git 仓库、依赖管理
- [x] **Day 2**: 评估框架 (Recall@K / MRR / NDCG@K) + 17 个 pytest 单元测试
- [x] **Day 3**: BEIR SciFact 数据加载 + BM25 检索器 (sklearn 风格 API) + 端到端 baseline
- [ ] **Day 4-6**: Dense 检索器 (Sentence-Transformers + FAISS)
- [ ] **Day 7-10**: Hybrid 融合 (Reciprocal Rank Fusion, k=60)
- [ ] **Day 11-14**: Cross-encoder 重排序 + 消融实验
- [ ] **Day 15-18**: 端到端 RAG (GPT-4o-mini) + RAGAS 评估生成质量
- [ ] **Day 19-21**: 错误案例分析、文档完善、对外发布

---

## 设计决策记录

**为什么不用 LangChain / LlamaIndex？**
框架封装会跳过 BM25 分词、IDF 计算、RRF 融合等关键细节。本项目从底层手写每一层，目标是建立对 RAG 系统的完整心智模型，而非熟练调用 API。

**为什么 BM25 索引时拼接 title + text？**
BEIR / SciFact 的 title 是高密度信号（每词都关键），仅对 text 建索引会丢失 title 关键词。这是 BEIR 论文官方 baseline 的标准做法。

**为什么用 RRF 融合而不是分数加权？**
RRF 公式 `1/(k+rank)` 只依赖排名不依赖分数，**无需跨模型分数归一化**，工程上最稳。

---

## License

MIT
