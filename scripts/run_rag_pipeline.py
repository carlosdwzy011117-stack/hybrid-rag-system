import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_scifact
from src.retrievers.dense_retriever import DenseRetriever
from src.generator import Generator


def run_pipeline():
    corpus, queries, qrels = load_scifact("data/scifact", split="test")

    # === 1. 实例化 DenseRetriever 并建索引 ===

    retriever = DenseRetriever()
    retriever.index(corpus)

    # === 2. 选一个 query 检索 ===

    qid = list(queries.keys())[0]
    query_text = queries[qid]

    # 调 search 拿 top-k。
    results = retriever.search(query_text, top_k=5)

    # === 3. 把 top-k 的 doc_id 映射回正文===

    doc_texts = [corpus[doc_id]["text"] for doc_id, _ in results]

    # === 4. 喂给 Generator 生成答案 ===

    gen = Generator()
    answer = gen.generate(query_text, doc_texts)
    return query_text, answer, doc_texts


if __name__ == "__main__":
    q, a, d = run_pipeline()
    print("=" * 50)
    print("Query:", q)
    print("Answer:", a)
    print("=" * 50)
