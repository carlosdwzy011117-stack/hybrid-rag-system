"""Smoke test for HybridRetriever using SciFact query 1 (known answer from Day 4)."""

import sys
from pathlib import Path

# 让 scripts/ 下脚本能 import src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_scifact
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.hybrid_retriever import HybridRetriever


def main():
    # ---------- 1. 加载数据 ----------
    data_dir = "data/scifact"
    corpus, queries, qrels = load_scifact(data_dir, split="test")
    print(f"[INFO] Loaded corpus={len(corpus)}, queries={len(queries)}, qrels={len(qrels)}")

    # ---------- 2. 构造 HybridRetriever 并建索引 ----------
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    hybrid = HybridRetriever(bm25, dense)   # k/pool_size 走默认
    hybrid.index(corpus)
    print("[INFO] Index built.")

    # ---------- 3. 拿 query 1 跑一下 ----------
    # queries 是 dict[qid -> text]，qid 是字符串，"1" 是已知的 query 1
    qid = "1"
    qtext = queries[qid]
    print(f"\n[INFO] Query 1 ({qid}): {qtext}")

    # ---------- 4. 跑 Hybrid search，看 gold doc 在哪 ----------
    # 已知 gold doc id = "31715818"（Day 4 + Day 6 都验过）
    GOLD_DOC_ID = "31715818"
    results = hybrid.search(qtext, top_k=10)

    # ---------- 5. 打印 top-10 + gold 位置 ----------
    for rank, (doc_id, score) in enumerate(results, start=1):
        mark = "  <-- GOLD" if doc_id == GOLD_DOC_ID else ""
        print(f"  #{rank}  doc={doc_id}  rrf={score:.4f}{mark}")

    gold_set = set(qrels[qid].keys())
    print(f"\n[INFO] Gold docs for this query: {gold_set}")



if __name__ == "__main__":
    main()