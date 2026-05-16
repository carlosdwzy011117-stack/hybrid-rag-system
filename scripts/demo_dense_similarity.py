"""Mini demo: 用 BGE-small 计算句子余弦相似度,直观感受 Dense 的语义匹配能力."""

from sentence_transformers import SentenceTransformer
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度.

    Args:
        a: 向量 1, shape (d,)
        b: 向量 2, shape (d,)

    Returns:
        余弦相似度, 范围 [-1, 1] (BGE 经过 L2 归一化后, 实际约 [0, 1])
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    print("Loading BGE-small...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    sentences = [
        "0-dimensional biomaterials show inductive properties.",
        "Nanoparticles can promote stem cell differentiation.",
        "The stock market crashed yesterday.",
    ]
    labels = ["A", "B", "C"]

    # 一次性 encode 3 句, 输出 shape (3, 384)
    embeddings = model.encode(sentences)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print("\nPairwise cosine similarity:")

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  {labels[i]} vs {labels[j]}: {sim:.4f}")


if __name__ == "__main__":
    main()