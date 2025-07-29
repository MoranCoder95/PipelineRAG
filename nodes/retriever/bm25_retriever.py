# nodes/retriever/bm25_retriever.py
from typing import Dict, List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from nodes.base import BaseNode
from storage.base import BaseStorage
from pipeline_data import PipelineData
import jieba


class BM25RetrieverNode(BaseNode):
    """BM25检索节点"""

    def __init__(self,
                 top_k: int = 3,
                 storage: Optional[BaseStorage] = None):
        """
        初始化BM25检索节点

        Args:
            top_k: 返回的最相关文档数量
            storage: 可选的存储实例
        """
        super().__init__(storage=storage)
        self.top_k = top_k
        self.bm25 = None
        self.docs = []

    def initialize_bm25(self, documents: List[Dict]):
        """初始化BM25索引"""
        self.docs = documents
        # 对文档进行分词
        tokenized_docs = [list(jieba.cut(doc["text"])) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def run(self, data: PipelineData, **kwargs) -> PipelineData:
        """
        执行BM25检索
        输入: data.content 应该是查询字符串
        输出: PipelineData 包含检索结果
        """
        query = data.content

        if self.bm25 is None:
            raise RuntimeError("检索器未初始化,请先提供文档构建索引")

        # 执行检索
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        # 获取top-k结果
        top_k_indices = np.argsort(scores)[-self.top_k:][::-1]
        results = []

        for idx in top_k_indices:
            doc = self.docs[idx]
            results.append({
                "text": doc["text"],
                "score": float(scores[idx]),
                "source": doc.get("source", ""),
                "metadata": doc.get("metadata", {})
            })

        result = PipelineData(
            content=results,
            metadata={
                'query': query,
                'total_docs': len(self.docs),
                'node_type': 'bm25_retrieval'
            },
            source='BM25Retriever'
        )

        self.save_output(result)
        return result