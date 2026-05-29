"""
scripts/scan_qids_for_answer.py
循环跑几条 SciFact query 的 pipeline,只打印 QID/Query/Answer。
肉眼扫哪条 Answer 是实质多句回答(非推脱),锁定它做 Day 16 重跑实验的难样本。
设计:build_pipeline() 建一次索引,answer_one() 循环复用 → 避免重复建索引。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run_rag_pipeline import build_pipeline, answer_one


def main():
    candidate_qids = ["1", "3", "5", "13", "36", "42", "48", "49"]
    print("[INFO] Building pipeline (index once)...")
    corpus, queries, _, retriever, gen = build_pipeline()
    print(f"[INFO] Done. Loaded queries={len(queries)}, corpus={len(corpus)}\n")
    for qid in candidate_qids:
        query_text, answer, _ = answer_one(qid, queries, corpus, retriever, gen)
        print(f"QID: {qid}")
        print(f"Query: {query_text}")
        print(f"Answer: {answer}")
        print("-" * 50)


if __name__ == "__main__":
    main()
