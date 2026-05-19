"""Sweep alpha hyperparameter for WeightedRetriever on SciFact."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_scifact
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.weighted_retriever import WeightedRetriever
from src.evaluator import recall_at_k, mrr, ndcg_at_k


def evaluate(retriever, queries, qrels, top_k=10):
    """Run a retriever over all queries and return (avg_recall, mrr_score, avg_ndcg)."""
    predicted_list = []
    gold_list = []
    for query_id, query_text in queries.items():
        results = retriever.search(query_text, top_k=top_k)
        predicted = [doc_id for doc_id, _ in results]
        gold = {doc_id for doc_id, score in qrels.get(query_id, {}).items() if score > 0}
        predicted_list.append(predicted)
        gold_list.append(gold)

    mrr_score = mrr(predicted_list, gold_list)
    recall_scores = [recall_at_k(p, g, k=top_k) for p, g in zip(predicted_list, gold_list)]
    avg_recall = sum(recall_scores) / len(recall_scores)
    ndcg_scores = [ndcg_at_k(p, g, k=top_k) for p, g in zip(predicted_list, gold_list)]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

    return avg_recall, mrr_score, avg_ndcg



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
        recall, mrr_score, ndcg = evaluate(weighted, queries, qrels)
        results_table.append((alpha, recall, mrr_score, ndcg))
        print(f"[alpha={alpha:.1f}] Recall@10={recall:.4f}  MRR={mrr_score:.4f}  NDCG@10={ndcg:.4f}")

    # ---------- 4. 打印对照表 ----------
    print("\n" + "=" * 60)
    print("Weighted Retriever Alpha Sweep on SciFact (n=300)")
    print("=" * 60)
    print(f"| {'alpha':<6} | {'Recall@10':<10} | {'MRR':<6} | {'NDCG@10':<8} |")
    print(f"|{'-' * 8}|{'-' * 12}|{'-' * 8}|{'-' * 10}|")
    for alpha, recall, mrr_score, ndcg in results_table:
        print(f"| {alpha:<6.1f} | {recall:<10.4f} | {mrr_score:<6.4f} | {ndcg:<8.4f} |")


if __name__ == "__main__":
    main()