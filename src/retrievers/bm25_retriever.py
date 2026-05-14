"""BM25 检索器。

基于 rank_bm25 库实现的简单 BM25 检索器，封装成统一接口。
"""

from rank_bm25 import BM25Okapi
import numpy as np


def _tokenize(text: str) -> list[str]:
    """简单分词：小写化 + 按空格切分。

    Args:
        text: 输入文本。

    Returns:
        分词后的 token 列表。

    Example:
        _tokenize("Vitamin D Helps")  -> ["vitamin", "d", "helps"]
    """
    # TODO: 你来填（1 行代码：lower() + split()）
    return text.lower().split()


class BM25Retriever:
    """BM25 检索器。

    用法：
        retriever = BM25Retriever()
        retriever.index(corpus)
        results = retriever.search("vitamin D", top_k=10)
    """

    def __init__(self):
        """初始化空的 retriever。

        建索引之前 self.bm25 和 self.doc_ids 都是 None。
        """
        self.doc_ids = None
        self.bm25 = None


    def index(self, corpus: dict) -> None:
        """对 corpus 建立 BM25 索引。

        Args:
            corpus: dict, doc_id -> {"text": str, "title": str}。
                例如来自 load_scifact 的第一个返回值。

        Side effects:
            填充 self.bm25 (BM25Okapi 对象) 和 self.doc_ids (list[str])。
        """
        doc_ids = []
        tokenized_corpus = []
        for doc_id, doc in corpus.items():
            text_to_tokenize = doc["title"] + " " + doc["text"]
            tokens = _tokenize(text_to_tokenize)
            doc_ids.append(doc_id)
            tokenized_corpus.append(tokens)
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.doc_ids = doc_ids

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """对一条 query 检索 top-K 文档。

        Args:
            query: 查询文本。
            top_k: 返回前 K 个文档，默认 10。

        Returns:
            list of (doc_id, score)，按 score 降序排列。

        Raises:
            RuntimeError: 当还没调用 index() 时。
        """
        # 防御：未建索引就调用 search 报错
        if self.bm25 is None:
            raise RuntimeError("必须先调用 index() 建索引再 search")


        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_indices:
            doc_id = self.doc_ids[i]
            score = float(scores[i])
            results.append((doc_id, score))
        return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.data_loader import load_scifact

    corpus, queries, qrels = load_scifact("data/scifact")
    retriever = BM25Retriever()
    retriever.index(corpus)
    first_qid = list(queries.keys())[0]
    query = queries[first_qid]
    gold_doc_ids = list(qrels[first_qid].keys())
    print(f"query: {query}")
    print(f"gold doc_ids: {gold_doc_ids}")
    print(f"\nBM25 top-5:")

    results = retriever.search(query, top_k=5)
    for doc_id, score in results:
        hit = "✓" if doc_id in gold_doc_ids else " "
        print(f"{hit} doc_id: {doc_id} score: {score:.3f}")