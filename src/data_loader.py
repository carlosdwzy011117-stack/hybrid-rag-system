"""SciFact 数据加载模块。

封装 BEIR 的 GenericDataLoader，提供统一的数据加载接口。
"""

from beir.datasets.data_loader import GenericDataLoader


def load_scifact(data_dir: str, split: str = "test"):
    """加载 SciFact 数据集。

    Args:
        data_dir: SciFact 数据所在目录，目录下应包含 corpus.jsonl、
            queries.jsonl 和 qrels/ 子目录。例如 "data/scifact"。
        split: 要加载的数据 split，默认 "test"。SciFact 有 train/dev/test
            三个 split，做检索 baseline 一般用 test。

    Returns:
        三元组 (corpus, queries, qrels):
            corpus  : dict, doc_id -> {"text": str, "title": str}
            queries : dict, query_id -> query_text (str)
            qrels   : dict, query_id -> {doc_id: relevance_score}

    Raises:
        FileNotFoundError: 当 data_dir 不存在或缺少必需文件时。

    Example:
        corpus, queries, qrels = load_scifact("data/scifact")
        -> corpus 有 5183 个文档
        -> queries 有 300 条查询
        -> qrels 有 300 条标注
    """

    corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split=split)
    return corpus, queries, qrels


if __name__ == "__main__":

    corpus, queries, qrels = load_scifact("data/scifact")

    print(f"corpus 文档数: {len(corpus)}")
    print(f"queries 数: {len(queries)}")
    print(f"qrels 标注数: {len(qrels)}")

    first_qid = list(queries.keys())[0]
    print(f"\n样本 query_id: {first_qid}")
    print(f"样本 query 文本: {queries[first_qid]}")
    print(f"样本相关文档: {list(qrels[first_qid].keys())}")