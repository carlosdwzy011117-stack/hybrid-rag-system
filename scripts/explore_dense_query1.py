"""探索性实验: BGE-small 对 SciFact query 1 的检索结果.

Day 3 已知: BM25 在 query 1 上未能召回 gold doc 31715818 进 top-5.
本实验验证: 同一条 query 在 BGE 下, gold doc 排名第几?
"""

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_scifact


def main():
    # ========================================
    # Step 1: 加载数据 + 取 query 1 + gold doc
    # ========================================
    corpus, queries, qrels = load_scifact("data/scifact", split="test")

    # 取出第一条 query (queries 是 dict, next(iter()) 拿第一对 key-value)
    query_id, query_text = next(iter(queries.items()))

    # 取该 query 对应的 gold doc_ids (qrels 里 score > 0 视为相关)
    gold = {doc_id for doc_id, score in qrels.get(query_id, {}).items() if score > 0}

    print(f"Query ID: {query_id}")
    print(f"Query text: {query_text}")
    print(f"Gold docs: {gold}")
    print(f"Corpus size: {len(corpus)}")

    # ========================================
    # Step 2: 准备 corpus 文本 (title + text 拼接, 和 BM25 对齐)
    # ========================================
    # 关键: 保持 doc_id 和文本的顺序对应, 后面要用 index 反查 doc_id
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did]["title"] + " " + corpus[did]["text"] for did in doc_ids]
    print(f"\nPrepared {len(doc_texts)} documents.")

    # ========================================
    # Step 3: BGE encode (corpus + query)
    # ========================================
    print("\nLoading BGE-small...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    print("Encoding corpus (this may take 2-3 minutes)...")
    doc_embeddings = model.encode(doc_texts, batch_size=32, show_progress_bar=True)

    print(f"Doc embeddings shape: {doc_embeddings.shape}")

    # encode 单条 query (输入是 str, 输出 shape (384,))
    query_embedding = model.encode(query_text)

    print(f"Query embedding shape: {query_embedding.shape}")

    # ========================================
    # Step 4: 算余弦相似度 + 排序
    # ========================================
    # 关键观察: BGE 输出已 L2 归一化 -> 余弦 = 点积
    # 一次矩阵乘法搞定 5183 篇文档相似度

    similarities = doc_embeddings @ query_embedding

    print(f"\nSimilarities shape: {similarities.shape}")
    print(f"Sim range: min={similarities.min():.4f}, max={similarities.max():.4f}")

    # 按相似度降序排序, 拿到排序后的 index 数组 (np.argsort 默认升序, [::-1] 反转)
    sorted_indices = np.argsort(similarities)[::-1]

    # ========================================
    # Step 5: 看 gold doc 的排名 + 看 top-5
    # ========================================
    # 5.a: 找 gold doc 在排序里的位置
    print("\n=== Gold doc 排名 ===")
    for gold_doc_id in gold:
        # 在 doc_ids 里找 gold_doc_id 的 index
        # 然后在 sorted_indices 里找这个 index 的位置 (rank)
        if gold_doc_id in doc_ids:
            doc_index = doc_ids.index(gold_doc_id)
            # np.where 返回元组, [0][0] 取第一个匹配位置
            rank = int(np.where(sorted_indices == doc_index)[0][0]) + 1
            score = float(similarities[doc_index])
            print(f"  Gold doc {gold_doc_id}: rank #{rank} / 5183, similarity={score:.4f}")
        else:
            print(f"  Gold doc {gold_doc_id} NOT FOUND in corpus!")

    # 5.b: 打印 BGE 的 top-5 结果
    print("\n=== BGE top-5 ===")
    for rank_pos, idx in enumerate(sorted_indices[:5], start=1):
        doc_id = doc_ids[idx]
        sim = similarities[idx]
        is_gold = "✅ GOLD" if doc_id in gold else ""
        title = corpus[doc_id]["title"][:80]
        print(f"  #{rank_pos} [{doc_id}] sim={sim:.4f} {is_gold}")
        print(f"      title: {title}")


if __name__ == "__main__":
    main()