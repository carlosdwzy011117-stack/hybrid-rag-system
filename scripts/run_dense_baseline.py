"""End-to-end Dense (BGE-small-en-v1.5) baseline on SciFact: load → encode → FAISS search → evaluate."""

# 标准库 import
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_scifact
from src.retrievers.dense_retriever import DenseRetriever
from src.evaluator import evaluate_retriever


def main():
    """Run Dense baseline on SciFact and print metrics."""

    # Step 1: 加载数据

    corpus, queries, qrels = load_scifact("data/scifact", split="test")

    # Step 2: 建 Dense 索引

    print("[INFO] Building dense index (BGE encoding ~30s)...")
    retriever = DenseRetriever()
    retriever.index(corpus)
    print("[INFO] Index built.")

    missing = [qid for qid in queries if qid not in qrels]
    if missing:
        print(f"[WARN] {len(missing)} queries 没有 qrels 标注: {missing[:5]}")

    # Step 3: 评估(evaluate_retriever 计算 K=1/5/10/20)

    metrics = evaluate_retriever(retriever, queries, qrels)
    print(f"Dense (BGE-small) Baseline on SciFact (n={len(queries)} queries):")
    print()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
