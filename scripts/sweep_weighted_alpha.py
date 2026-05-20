"""Sweep alpha hyperparameter for WeightedRetriever on SciFact."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_scifact
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.weighted_retriever import WeightedRetriever
from src.evaluator import evaluate_retriever


def main():
    # ---------- 1. 加载数据 ----------
    print("[INFO] Loading SciFact...")
    corpus, queries, qrels = load_scifact("data/scifact", split="test")
    print(f"[INFO] Loaded corpus={len(corpus)}, queries={len(queries)}")

    # ---------- 2. 建索引（只建一次！） ----------
    print("[INFO] Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.index(corpus)
    print("[INFO] Building Dense index (BGE encoding ~30s)...")
    dense = DenseRetriever()
    dense.index(corpus)
    print("[INFO] Both indexes built.\n")

    # ---------- 3. 扫描 alpha ----------
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results_table = []  # 存 (alpha, recall, mrr, ndcg) 元组

    for alpha in alphas:
        weighted = WeightedRetriever(bm25, dense, alpha=alpha)
        # 注意：不调 weighted.index()，bm25/dense 已建好索引
        metrics = evaluate_retriever(weighted, queries, qrels)
        results_table.append((alpha, metrics))
        # 提示：从 metrics dict 取值，key 是 "recall@10" / "recall@20" / "mrr" / "ndcg@20"
        print(f"[alpha={alpha:.1f}] Recall@10={metrics['recall@10']:.4f}  Recall@20={metrics['recall@20']:.4f}  MRR={metrics['mrr']:.4f}  NDCG@20={metrics['ndcg@20']:.4f}")
    # ---------- 4. 打印对照表 ----------
    print("\n" + "=" * 60)
    print("Weighted Retriever Alpha Sweep on SciFact (n=300)")
    print("=" * 60)
    print(f"| {'alpha':<6} | {'Recall@10':<10} | {'Recall@20':<10} | {'MRR':<6} | {'NDCG@20':<8} | ")
    print(f"|{'-' * 8}|{'-' * 12}|{'-' * 12}|{'-' * 8}|{'-' * 10}|")
    for alpha, metrics in results_table:
        print(f"| {alpha:<6.1f} | {metrics['recall@10']:<10.4f} | {metrics['recall@20']:<10.4f} | {metrics['mrr']:<6.4f} | {metrics['ndcg@20']:<8.4f} |")


if __name__ == "__main__":
    main()