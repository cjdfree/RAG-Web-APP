from __future__ import annotations

import logging
from typing import Dict, List, Any


from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)

class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """`Zhipuai Embeddings` embedding models."""

    client: Any
    """`zhipuai.ZhipuAI"""
    
    """
        root_validator 是 Pydantic 模块中一个用于自定义数据校验的装饰器函数。root_validator 用于在校验整个数据模型之前对整个数据模型进行自定义校验，以确保所有的数据都符合所期望的数据结构。
        root_validator 接收一个函数作为参数，该函数包含需要校验的逻辑。函数应该返回一个字典，其中包含经过校验的数据。如果校验失败，则抛出一个 ValueError 异常。
    """
    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """
        实例化ZhipuAI为values["client"]

        Args:

            values (Dict): 包含配置信息的字典，必须包含 client 的字段.
        Returns:

            values (Dict): 包含配置信息的字典。如果环境中有zhipuai库，则将返回实例化的ZhipuAI类；否则将报错 'ModuleNotFoundError: No module named 'zhipuai''.
        """
        from zhipuai import ZhipuAI
        values["client"] = ZhipuAI()
        return values
    
    # embed_query 是对单个文本（str）计算 embedding 的方法，这里我们重写该方法，调用验证环境时实例化的`ZhipuAI`来 调用远程 API 并返回 embedding 结果。
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.embeddings.create(
            model="embedding-2",
            input=text
        )
        return embeddings.data[0].embedding
    
    # 对文本的词向量封装，就是重复一句一句话来封装，调用embed_query的一句话的方法，重复使用就行
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        return [self.embed_query(text) for text in texts]
    
    # 下面两个异步的方法代码暂时不支持
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError("Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError("Please use `aembed_query`. Official does not support asynchronous requests")