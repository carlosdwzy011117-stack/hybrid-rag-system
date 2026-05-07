# Hybrid RAG System

## Status
🚧 Work in progress (started May 2026, exp. completion: late May 2026)

## Goal
Build an end-to-end retrieval-augmented generation (RAG) system on standard 
information retrieval benchmarks, combining sparse and dense retrieval with 
reranking and LLM-based answer generation.

## Planned Architecture
1. **Sparse Retrieval**: BM25 (rank_bm25)
2. **Dense Retrieval**: sentence-transformers (BGE / MPNet)
3. **Hybrid Fusion**: Reciprocal Rank Fusion (RRF)
4. **Reranking**: Cross-encoder
5. **Generation**: GPT-4o-mini with citation grounding
6. **Evaluation**: Recall@K, NDCG@10 (retrieval), RAGAS (generation)

## Datasets
[BEIR](https://github.com/beir-cellar/beir): SciFact

## Tech Stack
Python, sentence-transformers, FAISS, rank_bm25, OpenAI API

## Roadmap
- [x] Day 1: Environment setup, data loading
- [ ] Week 1: BM25 + Dense baselines
- [ ] Week 2: Hybrid fusion + Reranking + Ablation study
- [ ] Week 3: LLM generation + Demo