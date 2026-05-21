"""
evaluator 模块的单元测试
"""
import sys
import os
# 让 Python 能找到 src/ 目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import pytest
from src.evaluator import recall_at_k, reciprocal_rank, mrr, ndcg_at_k, evaluate_retriever


# ========== recall_at_k 测试 ==========

class TestRecallAtK:
    def test_all_hits(self):
        """所有 gold 都在 top-k 里"""
        assert recall_at_k(["d1", "d2", "d3"], {"d2"}, k=3) == 1.0

    def test_partial_hits(self):
        """部分 gold 在 top-k 里"""
        assert recall_at_k(["d1", "d2", "d3"], {"d2", "d5"}, k=3) == 0.5

    def test_no_hits(self):
        """完全没命中"""
        assert recall_at_k(["d1", "d2", "d3"], {"d9"}, k=3) == 0.0

    def test_empty_gold(self):
        """边界：gold 为空"""
        assert recall_at_k(["d1", "d2", "d3"], set(), k=3) == 0.0

    def test_k_smaller_than_hits(self):
        """k 小，部分命中被截断"""
        # gold={"d3"} 但 k=2，d3 在第 3 位，被截断
        assert recall_at_k(["d1", "d2", "d3"], {"d3"}, k=2) == 0.0


# ========== reciprocal_rank 测试 ==========

class TestReciprocalRank:
    def test_first_position(self):
        """第 1 位命中"""
        assert reciprocal_rank(["d1", "d2", "d3"], {"d1"}) == 1.0

    def test_third_position(self):
        """第 3 位命中"""
        # 用 pytest.approx 处理浮点数比较
        assert reciprocal_rank(["d1", "d2", "d3"], {"d3"}) == pytest.approx(1/3)

    def test_no_hit(self):
        """没命中"""
        assert reciprocal_rank(["d1", "d2", "d3"], {"d9"}) == 0.0

    def test_empty_gold(self):
        """边界：gold 为空"""
        assert reciprocal_rank(["d1", "d2", "d3"], set()) == 0.0

    def test_only_first_hit_counts(self):
        """有多个 gold，只看第一个命中位置"""
        # d1 在第 1 位（命中 gold），d3 在第 3 位（也命中），只看 d1
        assert reciprocal_rank(["d1", "d2", "d3"], {"d1", "d3"}) == 1.0


# ========== mrr 测试 ==========

class TestMRR:
    def test_multiple_queries(self):
        """多个 query 的 MRR"""
        # query1: 第 3 位命中 = 1/3
        # query2: 第 1 位命中 = 1.0
        # MRR = (1/3 + 1) / 2 = 2/3
        result = mrr(
            [["d1", "d2", "d3"], ["d4", "d5"]],
            [{"d3"}, {"d4"}]
        )
        assert result == pytest.approx(2/3)

    def test_empty_input(self):
        """边界：空输入"""
        assert mrr([], []) == 0.0

    def test_all_miss(self):
        """所有 query 都没命中"""
        result = mrr(
            [["d1", "d2"], ["d3", "d4"]],
            [{"d9"}, {"d8"}]
        )
        assert result == 0.0


# ========== ndcg_at_k 测试 ==========

class TestNDCGAtK:
    def test_handcalc_example(self):
        """手算验证的例子"""
        # gold={d2,d4}，d2 在第 2 位，d4 在第 4 位
        # NDCG@5 = (1/log2(3) + 1/log2(5)) / (1/log2(2) + 1/log2(3))
        result = ndcg_at_k(["d1","d2","d3","d4","d5"], {"d2","d4"}, k=5)
        assert result == pytest.approx(0.6509, abs=1e-3)

    def test_perfect_ranking(self):
        """完美排序：所有 gold 排在最前"""
        result = ndcg_at_k(["d2","d4","d1","d3","d5"], {"d2","d4"}, k=5)
        assert result == pytest.approx(1.0)

    def test_no_hit(self):
        """没命中"""
        assert ndcg_at_k(["d1", "d2", "d3"], {"d9"}, k=3) == 0.0

    def test_empty_gold(self):
        """边界：gold 为空"""
        assert ndcg_at_k(["d1", "d2", "d3"], set(), k=3) == 0.0


# ========== evaluate_retriever 测试 ==========

class TestEvaluateRetriever:
    """测 evaluate_retriever：组装 recall/mrr/ndcg 的上层函数。
    用假 retriever（写死 search 返回值），喂已知数据，对手算答案。
    """

    def test_single_query(self):
        """单 query，手算验证 recall/mrr。"""

        # 假 retriever：search 永远返回 d1, d2, d3（不管 query 是什么）
        class FakeRetriever:
            def search(self, query_text, top_k):
                return [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)]

        # 假数据：1 条 query，标准答案说 d2 相关
        queries = {"q1": "anything"}
        qrels = {"q1": {"d2": 1}}

        # 跑被测函数，只看 k=1 和 k=3 两个深度
        metrics = evaluate_retriever(FakeRetriever(), queries, qrels, k_values=(1, 3))
        assert metrics["recall@1"] == 0.0
        assert metrics["mrr"] == 0.5
        assert metrics["recall@3"] == 1.0
