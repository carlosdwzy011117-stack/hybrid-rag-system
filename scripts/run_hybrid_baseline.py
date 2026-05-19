"""End-to-end Hybrid (BM25 + Dense via RRF) baseline on SciFact: load → encode → FAISS search → evaluate."""

# 标准库 import
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_scifact
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.evaluator import evaluate_retriever


def main():
    """Run Hybrid baseline on SciFact and print metrics."""

    # Step 1: 加载数据

    corpus, queries, qrels = load_scifact("data/scifact", split="test")

    # Step 2: 建 Hybrid 索引

    print("[INFO] Building hybrid index (BM25 + Dense BGE)...")
    retriever = HybridRetriever(BM25Retriever(), DenseRetriever())
    retriever.index(corpus)
    print("[INFO] Index built.")

    # Sanity check: 确认每条 query 都有 qrels 标注
    missing = [qid for qid in queries if qid not in qrels]
    if missing:
        print(f"[WARN] {len(missing)} queries 没有 qrels 标注: {missing[:5]}")

    # Step 3: 用 evaluate_retriever 统一评估 (multi-K Recall + MRR + NDCG@max_K)
    metrics = evaluate_retriever(retriever, queries, qrels)

    # Step 4: 打印结果
    print(f"Hybrid (BM25 + Dense, RRF k=60, pool=100) Baseline on SciFact (n={len(queries)} queries):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
