"""
评估指标模块
-----------
提供信息检索任务的标准评估指标：Recall@K、MRR、NDCG@K

所有指标遵循统一约定：
- predicted: List[str]，检索系统返回的文档 ID 列表，按相关性降序
- gold: Set[str]，真实相关文档的 ID 集合
- 当 gold 为空时，所有指标返回 0.0
"""
import math
from typing import Dict, List, Set, Tuple


def recall_at_k(predicted: List[str], gold: Set[str], k: int) -> float:
    """前 K 个预测里命中 gold 的比例。"""
    if len(gold) == 0:
        return 0.0
    pre_set = set(predicted[:k])
    intersection = pre_set & gold
    return len(intersection) / len(gold)


def reciprocal_rank(predicted: List[str], gold: Set[str]) -> float:
    """单 query 的 Reciprocal Rank：1 / 第一个命中位置。"""
    if len(gold) == 0:
        return 0.0
    for i, doc in enumerate(predicted):
        if doc in gold:
            return 1 / (i + 1)
    return 0.0


def mrr(predicted_list: List[List[str]], gold_list: List[Set[str]]) -> float:
    """多 query 的 Mean Reciprocal Rank。"""
    if len(predicted_list) == 0:
        return 0.0
    assert len(predicted_list) == len(gold_list), \
        "predicted_list 和 gold_list 长度必须相同"

    rrs = [reciprocal_rank(p, g) for p, g in zip(predicted_list, gold_list)]
    return sum(rrs) / len(rrs)


def ndcg_at_k(predicted: List[str], gold: Set[str], k: int) -> float:
    """单 query 的 NDCG@K = DCG@K / IDCG@K。"""
    if len(gold) == 0:
        return 0.0

    # DCG: 实际排序的折损累积增益
    dcg = 0.0
    for i, doc in enumerate(predicted[:k]):
        if doc in gold:
            dcg += 1 / math.log2(i + 2)

    # IDCG: 理想排序的最大 DCG
    ideal_hits = min(len(gold), k)
    idcg = 0.0
    for i in range(ideal_hits):
        idcg += 1 / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retriever(
        retriever,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        k_values: Tuple[int, ...] = (1, 5, 10, 20),
) -> Dict[str, float]:
    """
    Run a retriever over all queries and compute Recall@K (multiple K), MRR, NDCG@max_K.

    Pulls retriever.search(query, top_k=max(k_values)) once per query, then
    slices the result list for each K. Empty-gold queries are kept and counted
    as 0 (consistent with src.evaluator.recall_at_k behavior).

    Args:
        retriever: object with .search(query_text, top_k) -> list of (doc_id, score)
        queries: dict[query_id -> query_text]
        qrels: dict[query_id -> dict[doc_id -> relevance_int]]
        k_values: K values for Recall@K. Largest is used as retrieval top_k
                  and for the single NDCG metric.

    Returns:
        dict, e.g. {
            "recall@1": 0.42, "recall@5": 0.71, "recall@10": 0.85, "recall@20": 0.91,
            "mrr": 0.69,
            "ndcg@20": 0.74,
        }
    """
    # ─── 准备 ───
    max_k = max(k_values)

    # ─── 主循环: 跑 retriever, 收集每个 query 的 predicted 和 gold ───
    predicted_list = []
    gold_list = []
    for query_id, query_text in queries.items():
        results = retriever.search(query_text, top_k=max_k)
        predicted = [doc_id for doc_id, _ in results]
        gold = {doc_id for doc_id, score in qrels.get(query_id, {}).items() if score > 0}
        predicted_list.append(predicted)
        gold_list.append(gold)

    # ─── 计算指标 ───
    metrics = {}

    for k in k_values:
        recall_scores = [recall_at_k(p, g, k=k) for p, g in zip(predicted_list, gold_list)]
        metrics[f"recall@{k}"] = sum(recall_scores) / len(recall_scores)

    metrics["mrr"] = mrr(predicted_list, gold_list)
    ndcg_scores = [ndcg_at_k(p, g, k=max_k) for p, g in zip(predicted_list, gold_list)]
    metrics[f"ndcg@{max_k}"] = sum(ndcg_scores) / len(ndcg_scores)

    return metrics
