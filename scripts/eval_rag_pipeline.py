import sys
import os
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ragas import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from run_rag_pipeline import build_pipeline, answer_one
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper


def run_eval_once(dataset, evaluator_llm, evaluator_embeddings):
    """跑一次 RAGAS evaluate，返回这一次的 {faithfulness, answer_relevancy} 数值。

    把原来那行 evaluate(...) 搬进来，然后从 result 里抽出两个指标的数值，
    打包成一个普通 dict 返回（方便外面 append 进 list）。
    """

    result = evaluate(dataset=dataset, embeddings=evaluator_embeddings,
                      metrics=[Faithfulness(), AnswerRelevancy(strictness=1)], llm=evaluator_llm)
    return {'faithfulness': float(result['faithfulness'][0]), 'answer_relevancy': float(result['answer_relevancy'][0])}


def run_eval_n_times(dataset, evaluator_llm, evaluator_embeddings, n=5):
    """跑 n 次评估，收集每次结果。

    返回一个 list，里面 n 个 dict，每个是一次 run_eval_once 的结果。
    """
    results = []

    for i in range(n):
        a = run_eval_once(dataset, evaluator_llm, evaluator_embeddings)
        results.append(a)
        print(f"第{i + 1}次: {a}")
    return results


def summarize(results):
    """从 n 次结果里，对每个指标算 mean 和 sample std。

    results 是 list[dict]，每个 dict 有 'faithfulness' 和 'answer_relevancy'。
    返回 {'faithfulness': {'mean': ..., 'std': ...}, 'answer_relevancy': {...}}
    """
    summary = {}
    for metric in results[0].keys():
        values = [r[metric] for r in results]
        summary[metric] = {'mean': statistics.mean(values), 'std': statistics.stdev(values)}
    return summary


# ===== 主流程 =====
if __name__ == '__main__':
    # ===== 只做一次的准备(循环外) =====
    corpus, queries, _, retriever, gen = build_pipeline()
    llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com")  # 旧代码搬过来
    evaluator_llm = LangchainLLMWrapper(llm)
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")  # 旧代码搬过来
    evaluator_embeddings = LangchainEmbeddingsWrapper(emb)

    # ===== 对每条难样本各重跑 5 次,分开统计 =====

    candidate_qids = ["49", "42"]

    for qid in candidate_qids:
        print(f"\n{'=' * 60}\n[QID {qid}] running 5x...\n{'=' * 60}")

        query, answer, docs = answer_one(qid, queries, corpus, retriever, gen)

        sample = SingleTurnSample(user_input=query, response=answer, retrieved_contexts=docs)
        dataset = EvaluationDataset(samples=[sample])

        results = run_eval_n_times(dataset, evaluator_llm, evaluator_embeddings, n=5)
        print(f"[QID {qid}] 每次结果:", results)

        summary = summarize(results)
        print(f"[QID {qid}] 汇总:", summary)
