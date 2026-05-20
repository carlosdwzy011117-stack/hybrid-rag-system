"""End-to-end Weighted (BM25 + Dense via min-max + weighted sum) baseline on SciFact: load → encode → FAISS search → evaluate."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_scifact
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.weighted_retriever import WeightedRetriever
from src.evaluator import evaluate_retriever


def main():
    """Run weighted baseline on SciFact and print metrics."""

    # Step 1: 加载数据

    corpus, queries, qrels = load_scifact("data/scifact", split="test")

    # Step 2: 建 weighted 索引

    print("[INFO] Building weighted index (BM25 + Dense BGE)...")
    retriever = WeightedRetriever(BM25Retriever(), DenseRetriever(), alpha=0.3)
    retriever.index(corpus)
    print("[INFO] Index built.")

    # Sanity check: 确认每条 query 都有 qrels 标注
    missing = [qid for qid in queries if qid not in qrels]
    if missing:
        print(f"[WARN] {len(missing)} queries 没有 qrels 标注: {missing[:5]}")

    # Step 3: 用 evaluate_retriever 统一评估 (multi-K Recall + MRR + NDCG@max_K)
    metrics = evaluate_retriever(retriever, queries, qrels)

    # Step 4: 打印结果
    print(f"Weighted (BM25 + Dense, alpha=0.3, pool=100) Baseline on SciFact (n={len(queries)} queries):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
