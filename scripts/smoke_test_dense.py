"""DenseRetriever 冒烟测试。

用 SciFact 的 query 1 验证类工作正常：
- gold doc 31715818 应该出现在 top-10
- 它的 similarity 应该接近 Day 4 探索的 0.7153
"""

import sys
from pathlib import Path



project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_scifact
from src.retrievers.dense_retriever import DenseRetriever


def main():
    # === 步骤 1：加载数据 ===
    # 📚 已写过同款：参照 scripts/run_bm25_baseline.py
    corpus, queries, qrels = load_scifact("data/scifact", split="test")

    # === 步骤 2：建 Dense 索引 ===
    print("[INFO] Building dense index (BGE encoding ~30s)...")
    retriever = DenseRetriever()
    retriever.index(corpus)
    print("[INFO] Index built.")

    # === 步骤 3：取第 1 条 query 和它的 gold ===
    # 📚 已写过同款：参照 BM25 baseline 末尾
    first_qid = list(queries.keys())[0]
    query = queries[first_qid]
    gold_doc_ids = list(qrels[first_qid].keys())

    print(f"\nquery: {query}")
    print(f"gold doc_ids: {gold_doc_ids}")

    # === 步骤 4：检索 top-10 ===
    results = retriever.search(query, top_k=10)

    print(f"\nDense top-10:")
    for rank, (doc_id, score) in enumerate(results, start=1):
        hit = "[HIT]" if doc_id in gold_doc_ids else "     "
        print(f"  {hit} #{rank}  doc_id: {doc_id}  score: {score:.4f}")

    # === 步骤 5：对照 Day 4 探索结果 ===
    found_gold = any(doc_id in gold_doc_ids for doc_id, _ in results)
    if found_gold:
        print("\n[SUCCESS] Gold doc found in top-10. Dense retriever working.")
    else:
        print("\n[WARN] Gold doc NOT in top-10. Day 4 showed it should be at #6.")


if __name__ == "__main__":
    main()

    