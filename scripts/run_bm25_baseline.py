"""End-to-end BM25 baseline on SciFact: load → index → search → evaluate."""

# 标准库 import
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_scifact
from src.retrievers.bm25_retriever import BM25Retriever
from src.evaluator import evaluate_retriever


def main():
    """Run BM25 baseline on SciFact and print metrics."""

    # Step 1: 加载数据

    corpus, queries, qrels = load_scifact("data/scifact", split="test")

    # Step 2: 建 BM25 索引

    retriever = BM25Retriever()
    retriever.index(corpus)

    # Sanity check: 确认每条 query 都有 qrels 标注
    missing = [qid for qid in queries if qid not in qrels]
    if missing:
        print(f"[WARN] {len(missing)} queries 没有 qrels 标注: {missing[:5]}")
    # Step 3: 用 evaluate_retriever 统一评估 (multi-K Recall + MRR + NDCG@max_K)
    metrics = evaluate_retriever(retriever, queries, qrels)

    # Step 4: 打印结果
    print(f"BM25 Baseline on SciFact (n={len(queries)} queries):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
