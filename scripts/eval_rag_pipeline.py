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


def compute_within_group_variance(results, metric_name):
    """组内方差：每个 qid 5 次重跑的方差，再对所有 qid 求平均。

    度量"裁判 LLM 的不稳定性"——同一 query 同一答案，多次评估分数应该一样，
    波动只来自 LLM 的内在随机性。

    Args:
        results: 嵌套 dict {qid: list[dict]}，每个 dict 含 'faithfulness' 等 metric
        metric_name: 'faithfulness' 或 'answer_relevancy'

    Returns:
        float: 组内方差均值（用 sample variance ddof=1）
    """
    variances = []
    for qid, runs in results.items():
        values = [r[metric_name] for r in runs]
        variance = statistics.variance(values)
        variances.append(variance)
    return statistics.mean(variances)


def compute_between_group_variance(results, metric_name):
    """组间方差：每个 qid 的 5 次重跑均值，再对这些均值算方差。

    度量"样本难度的真实差异"——不同 query 本身就该有不同的真实分数。

    Args:
        results: 同上
        metric_name: 同上

    Returns:
        float: 组间方差 (用 sample variance ddof=1)
    """
    group_means = []
    for qid, runs in results.items():
        values = [r[metric_name] for r in runs]
        group_mean = statistics.mean(values)
        group_means.append(group_mean)
    return statistics.variance(group_means)


# ===== 主流程 =====
if __name__ == '__main__':
    # ===== 只做一次的准备(循环外) =====
    corpus, queries, _, retriever, gen = build_pipeline()
    llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com")  # 旧代码搬过来
    evaluator_llm = LangchainLLMWrapper(llm)
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")  # 旧代码搬过来
    evaluator_embeddings = LangchainEmbeddingsWrapper(emb)

    # ===== 对每条难样本各重跑 5 次,分开统计 =====

    candidate_qids = ["49", "124", "42", "57", "137"]
    nested_results = {}
    for qid in candidate_qids:
        print(f"\n{'=' * 60}\n[QID {qid}] running 5x...\n{'=' * 60}")
        query, answer, docs = answer_one(qid, queries, corpus, retriever, gen)

        sample = SingleTurnSample(user_input=query, response=answer, retrieved_contexts=docs)
        dataset = EvaluationDataset(samples=[sample])

        results = run_eval_n_times(dataset, evaluator_llm, evaluator_embeddings, n=5)
        nested_results[qid] = results
        print(f"[QID {qid}] 每次结果:", results)

        summary = summarize(results)
        print(f"[QID {qid}] 汇总:", summary)
    faithfulness_within = compute_within_group_variance(nested_results, "faithfulness")
    faithfulness_between = compute_between_group_variance(nested_results, "faithfulness")
    answer_relevancy_within = compute_within_group_variance(nested_results, "answer_relevancy")
    answer_relevancy_between = compute_between_group_variance(nested_results, "answer_relevancy")
    print("=== Variance Decomposition ===")
    print(f"faithfulness:     within={faithfulness_within:.4f}  between={faithfulness_between:.4f}")
    print(f"answer_relevancy: within={answer_relevancy_within:.4f}  between={answer_relevancy_between:.4f}")
