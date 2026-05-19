"""Hybrid retriever combining BM25 and Dense via Reciprocal Rank Fusion (RRF)."""

from typing import List, Tuple
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever


class HybridRetriever:
    """通过 RRF 融合 BM25 和 Dense 两路检索结果的混合检索器。

    RRF 公式：
        score(d) = Σ over retrievers r of  1 / (k + rank_r(d))
    rank 从 1 开始，k 是平滑常数（默认 60）。

    候选池中不存在的文档，该路贡献为 0（即 rank 视为无穷大）。

    设计说明
    --------
    - 每路候选池由 `pool_size` 控制（默认 100），避免某 doc 在 BM25 排
      第 50 但在 Dense 排第 3 时被截断。最终 top_k 在融合后再切片。
    - 通过组合（composition）持有 BM25Retriever 和 DenseRetriever 实例，
      保持与其他 retriever 一致的 sklearn 风格 API。
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        k: int = 60,
        pool_size: int = 100,
    ):
        """
        参数
        ----
        bm25 : BM25Retriever
            BM25 检索器实例（可建好索引也可未建）。
        dense : DenseRetriever
            Dense 检索器实例（可建好索引也可未建）。
        k : int
            RRF 平滑常数。论文经验值 60（Cormack et al. 2009）。
        pool_size : int
            每个子检索器拉多少候选进融合池。
        """
        self.bm25 = bm25
        self.dense = dense
        self.k = k
        self.pool_size = pool_size

    def index(self, corpus: List[Tuple[str, str, str]]) -> None:
        """在同一个 corpus 上同时为两路子检索器建索引。

        参数
        ----
        corpus : list of (doc_id, title, text)
            和 BM25Retriever / DenseRetriever 期望的格式一致。
        """

        self.bm25.index(corpus)
        self.dense.index(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """对一个 query 做 RRF 融合检索，返回 top-k。

        步骤
        ----
        1. 分别从 BM25 和 Dense 各拿 top-`pool_size` 候选。
        2. 遍历两路结果，按 rank 累加 RRF 分数到字典：
           score[doc_id] += 1 / (k + rank)，rank 从 1 开始。
        3. 按分数降序排序，切片 top_k，返回 [(doc_id, rrf_score), ...]。

        返回
        ----
        list of (doc_id, rrf_score)，长度 <= top_k
        """
        # ---------- Step 1：拿两路候选 ----------
        # 🔧 直接抄：调子 retriever 的 search，注意 top_k 传 self.pool_size
        bm25_results = self.bm25.search(query, top_k=self.pool_size)
        dense_results = self.dense.search(query, top_k=self.pool_size)

        # ---------- Step 2：RRF 累加 ----------
        scores = {}
        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)

        # ---------- Step 3：排序 + 截 top_k ----------
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
