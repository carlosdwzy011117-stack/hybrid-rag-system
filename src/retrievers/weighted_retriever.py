"""Weighted retriever combining BM25 and Dense via min-max normalization + weighted sum."""

from typing import List, Tuple
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever


class WeightedRetriever:
    """通过 min-max 归一化 + 加权和融合 BM25 和 Dense 的混合检索器。

    融合公式：
        score(d) = α · BM25_norm(d) + (1 - α) · Dense_norm(d)
    其中 BM25_norm / Dense_norm 是各路 top-`pool_size` 内的 min-max 归一化分数。
    候选池外的 doc 在该路按 0 计。

    设计说明
    --------
    - α 是可调权重，对 SciFact 这种 BM25 弱、Dense 强的数据集，α 应该偏小（如 0.2）。
    - min-max 归一化是 per-query 的（每条 query 的 top-`pool_size` 各自归一化）。
    - 缺席处理：某 doc 只在 BM25 候选池而不在 Dense 候选池时，Dense 归一分按 0 计，
      反之亦然（与 RRF "rank=∞ → 贡献 0" 思路一致）。
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        alpha: float = 0.3,
        pool_size: int = 100,
    ):
        """
        参数
        ----
        bm25 : BM25Retriever
            BM25 检索器实例。
        dense : DenseRetriever
            Dense 检索器实例。
        alpha : float
            BM25 权重，Dense 权重为 (1 - alpha)。默认 0.3。
        pool_size : int
            每个子检索器拉多少候选进融合池。
        """
        # 🔧 直接抄：4 个参数原样存
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha
        self.pool_size = pool_size

    def index(self, corpus: List[Tuple[str, str, str]]) -> None:
        """在同一个 corpus 上同时为两路子检索器建索引。"""
        # 🔧 直接抄：和 HybridRetriever.index 一模一样
        self.bm25.index(corpus)
        self.dense.index(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """对一个 query 做加权融合检索，返回 top-k。

        步骤
        ----
        1. 两路各拿 top-`pool_size` 候选。
        2. 每路分数做 min-max 归一化（per-query）。
        3. 合并候选池（取并集），对每个 doc 算加权和，缺席的一路按 0。
        4. 排序，截 top_k。

        返回
        ----
        list of (doc_id, weighted_score)，长度 <= top_k
        """
        # ---------- Step 1：拿两路候选 ----------
        # 🔧 直接抄：和 HybridRetriever 一样
        bm25_results = self.bm25.search(query, top_k=self.pool_size)
        dense_results = self.dense.search(query, top_k=self.pool_size)


        # ---------- Step 2：min-max 归一化 ----------
        def _minmax(results):
            """Min-max normalize a list of (doc_id, raw_score) to dict[doc_id -> norm]."""

            if not results:
                return {}
            scores = [s for _, s in results]
            min_s = min(scores)
            max_s = max(scores)
            if max_s == min_s:
                return {doc_id : 0.0 for doc_id, _ in results}
            return {doc_id: (s-min_s) / (max_s - min_s) for doc_id, s in results}

        bm25_norm = _minmax(bm25_results)
        dense_norm = _minmax(dense_results)

        # ---------- Step 3：合并候选池 + 加权和 ----------
        scores = {}
        all_doc_ids = set(bm25_norm) | set(dense_norm)
        for doc_id in all_doc_ids:
            b = bm25_norm.get(doc_id, 0.0)
            d = dense_norm.get(doc_id, 0.0)
            final = self.alpha * b + (1 - self.alpha) * d
            scores[doc_id] = final

        # ---------- Step 4：排序 + 截 top_k ----------
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]