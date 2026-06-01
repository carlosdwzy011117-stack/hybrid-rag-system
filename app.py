"""Streamlit demo for Hybrid RAG System on BEIR/SciFact.

启动方式：
    streamlit run app.py

依赖前置：
    - data/scifact/ 数据已就位
    - 环境变量 OPENAI_API_KEY 已设置
"""

import streamlit as st

from src.data_loader import load_scifact
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.weighted_retriever import WeightedRetriever
from src.generator import Generator


# ============================================================
# Streamlit 页面配置（必须是第一个 st.* 调用）
# ============================================================
st.set_page_config(
    page_title="Hybrid RAG Demo",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# 慢操作：加载数据 + 建索引，用 @st.cache_resource 缓存
# 关键：这个函数只在首次启动时跑一次，之后用户每次点按钮都直接拿缓存
# 不缓存的话每次点按钮都会重建 BM25 + 重新跑 BGE 编码 5K 文档，要等 30 秒
# ============================================================
@st.cache_resource(show_spinner="Loading data and building indices...")
def load_retrievers_and_generator():
    """加载 corpus + 建三个 retriever 的索引 + 实例化 Generator。

    Returns:
        dict: {
            "corpus": dict,
            "retrievers": {"BM25": ..., "Dense": ..., "Weighted (α=0.3)": ...},
            "generator": Generator,
        }
    """
    # 1. 加载数据
    corpus, queries, qrels = load_scifact("data/scifact", split="test")

    # 2. 建 BM25 索引
    bm25 = BM25Retriever()
    bm25.index(corpus)

    # 3. 建 Dense 索引（这一步最慢，BGE 编码 5K 文档大概 1-2 分钟）
    dense = DenseRetriever()
    dense.index(corpus)

    # 4. 构造 Weighted retriever
    # 注意：复用已建好的 bm25 + dense 实例，不再调 weighted.index()
    # 因为 WeightedRetriever.index() 会重复调用底层两路的 index() —— 在 demo 场景下浪费时间
    weighted = WeightedRetriever(bm25=bm25, dense=dense, alpha=0.3)

    # 5. 实例化 Generator（注意：__init__ 里只是配置 client，没真正调 API）
    generator = Generator()

    return {
        "corpus": corpus,
        "queries": queries,
        "retrievers": {
            "BM25": bm25,
            "Dense (BGE)": dense,
            "Weighted (α=0.3)": weighted,
        },
        "generator": generator,
    }


# ============================================================
# 加载（首次会显示 spinner，后续走缓存秒开）
# ============================================================
resources = load_retrievers_and_generator()
corpus = resources["corpus"]
queries = resources["queries"]
retrievers = resources["retrievers"]
generator = resources["generator"]


# ============================================================
# 页面标题区
# ============================================================
st.title("🔍 Hybrid RAG System on BEIR/SciFact")
st.markdown(
    "End-to-end RAG demo: **retrieval (BM25 / Dense / Weighted Fusion) → "
    "LLM answer generation**. Switch retrievers to see how results change."
)

# 简明结果表（让面试官第一眼就看到核心数字）
with st.expander("📊 Retrieval Benchmark Results (NDCG@10 on BEIR/SciFact)", expanded=False):
    st.markdown(
        """
        | Method | Recall@10 | MRR | NDCG@10 |
        |---|---|---|---|
        | BM25 | 0.6862 | 0.5242 | 0.5597 |
        | Dense (BGE-small) | 0.8452 | 0.6845 | 0.7200 |
        | Hybrid (RRF, k=60) | 0.7877 | 0.6283 | 0.6617 |
        | **Weighted Fusion (α=0.2)** | **0.8502** | **0.6934** | **0.7269** |
        | Weighted Fusion (α=0.3) | 0.8562 | 0.6850 | 0.7239 |

        Dataset: 5,183 docs · 300 test queries · NDCG@10 +29% over BM25 baseline.
        """
    )


# ============================================================
# 侧边栏：example queries（让面试官现场不用想 query）
# ============================================================
st.sidebar.header("💡 Example Queries")
st.sidebar.markdown("Click to try out sample queries:")

EXAMPLE_QUERIES = {
    "Q1 (BM25 fails, Dense saves)": "0-dimensional biomaterials show inductive properties.",
    "Q42 (contradict claim)": queries.get("42", "ALDH1 expression is associated with breast cancer."),
    "Q49 (support claim)": queries.get("49", "Lymphocytic choriomeningitis virus...")[:100],
    "Q124 (support claim)": queries.get("124", "Vitamin D supplementation reduces...")[:100],
}

for label, query_text in EXAMPLE_QUERIES.items():
    if st.sidebar.button(label, use_container_width=True):
        st.session_state["query_input"] = query_text


# ============================================================
# 主区域：query 输入 + 控件
# ============================================================
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Enter your query:",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g. Does vitamin D supplementation reduce respiratory infections?",
    )

with col2:
    method = st.selectbox(
        "Retrieval method:",
        list(retrievers.keys()),
        index=2,  # 默认选 Weighted
    )

top_k = st.slider("Number of top documents to retrieve (top-K):", min_value=3, max_value=10, value=5)

col_left, col_right = st.columns(2)
with col_left:
    do_generate = st.checkbox("🤖 Generate answer with LLM", value=True)
with col_right:
    search_button = st.button("🚀 Search & Generate", type="primary", use_container_width=True)


# ============================================================
# 检索 + 生成主逻辑
# ============================================================
if search_button:
    if not query.strip():
        st.warning("⚠️ Please enter a query first.")
        st.stop()

    # ----------- 检索 -----------
    retriever = retrievers[method]
    with st.spinner(f"Retrieving with {method}..."):
        results = retriever.search(query, top_k=top_k)

    st.subheader(f"📄 Top-{top_k} Retrieved Documents ({method})")

    if not results:
        st.error("No documents retrieved.")
        st.stop()

    # 展示检索结果
    for rank, (doc_id, score) in enumerate(results, start=1):
        doc = corpus.get(doc_id, {})
        title = doc.get("title", "(no title)")
        text = doc.get("text", "(no text)")

        # 用 expander 收起长文本，默认 top-3 展开
        with st.expander(
            f"**#{rank}** · doc_id=`{doc_id}` · score={score:.4f} · {title}",
            expanded=(rank <= 3),
        ):
            st.write(text)

    # ----------- LLM 生成 -----------
    if do_generate:
        st.subheader("💬 Generated Answer (DeepSeek)")
        doc_texts = [corpus[doc_id]["text"] for doc_id, _ in results if doc_id in corpus]

        with st.spinner("Generating answer with LLM..."):
            try:
                answer = generator.generate(query, doc_texts)
                st.success(answer)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                st.info(
                    "Check that OPENAI_API_KEY env var is set and your DeepSeek "
                    "account has balance."
                )


# ============================================================
# 页脚
# ============================================================
st.markdown("---")
st.markdown(
    "**Project**: [Hybrid RAG System on BEIR/SciFact]"
    "(https://github.com/carlosdwzy011117-stack/hybrid-rag-system) · "
    "BM25 · BGE-small · FAISS · Weighted Fusion · RAGAS · DeepSeek"
)