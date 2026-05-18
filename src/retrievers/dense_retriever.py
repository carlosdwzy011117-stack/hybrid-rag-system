"""Dense 检索器。

基于 sentence-transformers (BGE-small-en-v1.5) + FAISS IndexFlatIP 实现。
设计与 BM25Retriever 平行，统一 sklearn 风格 API。
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class DenseRetriever:
    """Dense 检索器。

    用法：
        retriever = DenseRetriever()
        retriever.index(corpus)
        results = retriever.search("vitamin D", top_k=10)

    内部细节：
        - 编码器：BGE-small-en-v1.5（384 维，L2 归一化输出）
        - 索引：FAISS IndexFlatIP（内积 = 余弦，因为 BGE 已归一化）
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """初始化 retriever，加载编码器模型。

        Args:
            model_name: HuggingFace 模型名，默认 BGE-small-en-v1.5。

        Note:
            模型加载是慢操作（首次几秒），所以放在 __init__ 而不是 index()。
        """
        self.model = SentenceTransformer(model_name)
        self.doc_ids = None
        self.faiss_index = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        """统一编码入口。

        Args:
            texts: 要编码的字符串列表。

        Returns:
            shape = (len(texts), dim) 的 float32 numpy array，已 L2 归一化。

        Note:
            FAISS 强制要求 float32（默认是 float64 会报错）。
            normalize_embeddings=True 保证内积 = 余弦。
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def index(self, corpus: dict) -> None:
        """对 corpus 建立 FAISS 索引。

        Args:
            corpus: dict, doc_id -> {"text": str, "title": str}。

        Side effects:
            填充 self.doc_ids (list[str]) 和 self.faiss_index (faiss.IndexFlatIP)。

        实现步骤：
            1. 把 corpus.keys() 固化成 self.doc_ids（顺序敏感！）
            2. 按 self.doc_ids 的顺序拼接 title + text，得到 list[str]
            3. 调 self._encode() 得到 (N, dim) 矩阵
            4. 用 embeddings.shape[1] 拿到 dim，创建 faiss.IndexFlatIP(dim)
            5. 把 embeddings 加进 index
        """

        doc_ids = list(corpus.keys())
        texts = [corpus[did]["title"] + " " + corpus[did]["text"] for did in doc_ids]
        embeddings = self._encode(texts)
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings)
        self.doc_ids = doc_ids
        self.faiss_index = faiss_index


    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """对一条 query 检索 top-K 文档。

        Args:
            query: 查询文本。
            top_k: 返回前 K 个文档，默认 10。

        Returns:
            list of (doc_id, score)，按 score 降序排列。

        Raises:
            RuntimeError: 当还没调用 index() 时。

        实现步骤：
            1. 防御性检查（self.faiss_index is None 就报错）
            2. 把 query 编码成 (1, dim) 矩阵（注意 reshape）
            3. 调 self.faiss_index.search(query_emb, top_k)
               返回 scores (1, top_k) 和 indices (1, top_k)
            4. 用 self.doc_ids[i] 把行号映射回 doc_id
            5. 组装 [(doc_id, float(score)), ...] 返回
        """
        if self.faiss_index is None:
            raise RuntimeError("必须先调用 index() 建索引再 search")
        query_emb = self._encode([query])
        scores, indices = self.faiss_index.search(query_emb, top_k)
        results = []
        for i, s in zip(indices[0], scores[0]):
            doc_id = self.doc_ids[i]
            score = float(s)
            results.append((doc_id, score))
        return results