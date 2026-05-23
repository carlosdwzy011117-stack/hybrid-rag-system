import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ragas import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from run_rag_pipeline import run_pipeline
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

query, answer, docs = run_pipeline()

print("Query:", query)
print("Answer:", answer)
print("Docs 类型:", type(docs), "篇数:", len(docs))

sample = SingleTurnSample(user_input=query, response=answer, retrieved_contexts=docs)
dataset = EvaluationDataset(samples=[sample])
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
evaluator_embeddings = LangchainEmbeddingsWrapper(emb)
llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com")
evaluator_llm = LangchainLLMWrapper(llm)
result = evaluate(dataset=dataset, embeddings=evaluator_embeddings,
                  metrics=[Faithfulness(), AnswerRelevancy(strictness=1)], llm=evaluator_llm)
print(result)
