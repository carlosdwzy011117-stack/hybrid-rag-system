import os
from openai import OpenAI


class Generator:
    """用 LLM 根据检索到的文档生成答案。Provider 可配置（OpenAI / DeepSeek 等）。"""

    def __init__(self, model="deepseek-chat", base_url="https://api.deepseek.com"):
        self.model = model

        # 建 OpenAI client，照抄即可（api_key 从环境变量读，base_url 用传入的参数）
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def generate(self, query, docs):
        """根据检索到的文档正文，回答 query。

        query: str，用户问题
        docs: list[str]，检索到的文档正文（已经是文本，不是 doc_id）
        return: str，LLM 生成的答案
        """
        #  第一步：把 docs 列表拼成一段 context 文本

        context = "\n\n".join(docs)

        #  第二步：构造 messages（list of dict），把 context 和 query 放进 user 消息

        messages = [{"role": "system","content": "你是一个问答助手，请根据提供的文档回答问题，如果文档中没有相关信息就回答不知道"},{"role":"user","content": f"参考文档：{context}\n\n问题：{query}"}]

        #  第三步：调 API（照抄）

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        #  第四步：从 response 取出答案文本并 return

        return response.choices[0].message.content
