from ragas import evaluate
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample, EvaluationDataset
# 目的：用假数据验证 RAGAS 的 SingleTurnSample / EvaluationDataset 能正确构造

sample = SingleTurnSample(user_input="猫是什么动物？", response="猫是一种小型哺乳动物",
                          retrieved_contexts=["猫是一种小型哺乳动物，通常作为宠物饲养",
                                              "狗是人类最忠诚的朋友，喜欢摇尾巴。"])
dataset = EvaluationDataset(samples=[sample])

# 打印出来确认结构
print("sample:", sample)
print("dataset length:", len(dataset))

# 建一个走 DeepSeek 的 LangChain ChatOpenAI 对象

llm = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com")

evaluator_llm = LangchainLLMWrapper(llm)


result = evaluate(dataset=dataset, llm=evaluator_llm, metrics=[Faithfulness()])

# 打印结果
print("result:", result)
