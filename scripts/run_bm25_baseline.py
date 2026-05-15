"""End-to-end BM25 baseline on SciFact: load → index → search → evaluate."""

# 标准库 import
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_scifact
from src.retrievers.bm25_retriever import BM25Retriever
from src.evaluator import recall_at_k, mrr, ndcg_at_k


def main():
    """Run BM25 baseline on SciFact and print metrics."""

    # Step 1: 加载数据

    corpus, queries, qrels = load_scifact("data/scifact", split="test")

    # Step 2: 建 BM25 索引

    retriever = BM25Retriever()
    retriever.index(corpus)

    # Step 3: 遍历 queries，构造 predicted_list 和 gold_list

    top_k = 10
    predicted_list = []   # 外层 list, 每元素是 [doc_id, doc_id, ...]
    gold_list = []        # 外层 list, 每元素是 {doc_id, doc_id, ...} (set)
    # Sanity check: 确认每条 query 都有 qrels 标注
    missing = [qid for qid in queries if qid not in qrels]
    if missing:
        print(f"[WARN] {len(missing)} queries 没有 qrels 标注: {missing[:5]}")
    for query_id, query_text in queries.items():
        # 3.a 构造 predicted: search 拿 top-K, 只要 doc_id 不要 score
        results = retriever.search(query_text, top_k=top_k)
        predicted = [doc_id for doc_id, score in results]

        # 3.b 构造 gold: 从 qrels[query_id] 取出 score>0 的 doc_id, 存成 set
        gold = {doc_id for doc_id, score in qrels.get(query_id, {}).items() if score > 0}

        # 3.c append 到外层 list
        predicted_list.append(predicted)
        gold_list.append(gold)



    # Step 4: 计算评估指标

    # mrr 是 list-of-list 入口, 直接传整体
    mrr_score = mrr(predicted_list, gold_list)

    # recall / ndcg 是单 query 版本, 循环每条算完再平均
    recall_scores = [recall_at_k(p, g, k=top_k) for p, g in zip(predicted_list, gold_list)]
    avg_recall = sum(recall_scores) / len(recall_scores)
    ndcg_scores = [ndcg_at_k(p, g, k=top_k) for p, g in zip(predicted_list, gold_list)]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

    # Step 5: 打印结果

    print(f"BM25 Baseline on SciFact (n={len(queries)} queries):")
    print(f"  Recall@{top_k}: {avg_recall:.4f}")
    print(f"  MRR:        {mrr_score:.4f}")
    print(f"  NDCG@{top_k}: {avg_ndcg:.4f}")


if __name__ == "__main__":
    main()
