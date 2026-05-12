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
from typing import List, Set


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