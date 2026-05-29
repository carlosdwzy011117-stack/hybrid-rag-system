import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_scifact
from src.retrievers.dense_retriever import DenseRetriever
from src.generator import Generator


def build_pipeline():
    """加载数据 + 建索引 + 实例化 generator,返回可复用的家当。
    只做一次的重活(建索引最慢),供 answer_one 反复使用。

    Returns:
        corpus, queries, qrels, retriever, gen
    """
    corpus, queries, qrels = load_scifact("data/scifact", split="test")
    retriever = DenseRetriever()
    retriever.index(corpus)
    gen = Generator()
    return corpus, queries, qrels, retriever, gen


def answer_one(qid, queries, corpus, retriever, gen):
    """用已建好的家当跑单条 query,返回三元组。

    Args:
        qid: 字符串 query id
        queries, corpus, retriever, gen: build_pipeline()
    Returns:
        query_text, answer, doc_texts
    """
    query_text = queries[qid]
    results = retriever.search(query_text, top_k=5)
    doc_texts = [corpus[doc_id]["text"] for doc_id, _ in results]
    answer = gen.generate(query_text, doc_texts)
    return query_text, answer, doc_texts


def run_pipeline(qid="1"):
    """便捷包装:建家当 + 跑单条 query,返回三元组。
    单条用途的入口;循环跑多条时直接用 build_pipeline + answer_one(避免重复建索引)。
    """
    corpus, queries, qrels, retriever, gen = build_pipeline()
    return answer_one(qid, queries, corpus, retriever, gen)


if __name__ == "__main__":
    q, a, d = run_pipeline(qid="36")
    print("=" * 50)
    print("Query:", q)
    print("Answer:", a)
    print("=" * 50)
