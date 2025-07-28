# # nodes/retriever/hybrid_retriever.py
# import numpy as np
# from typing import Dict, List, Optional, Tuple
# import logging
# from rank_bm25 import BM25Okapi
# from nodes.base import BaseComponent
# from storage.base import BaseStorage
# import jieba
#
# logger = logging.getLogger(__name__)
#
#
# class HybridRetrieverNode(BaseComponent):
#     """混合检索节点,结合向量检索和BM25检索"""
#
#     def __init__(self,
#                  vector_store_node,
#                  top_k: int = 3,
#                  alpha: float = 0.5,  # BM25和向量检索的权重因子
#                  storage: Optional[BaseStorage] = None):
#         """
#         初始化混合检索节点
#
#         Args:
#             vector_store_node: 向量存储节点实例
#             top_k: 返回的最相关文档数量
#             alpha: BM25分数的权重(0-1),剩余权重分配给向量检索分数
#             storage: 可选的存储实例
#         """
#         super().__init__(storage=storage)
#         self.vector_store = vector_store_node
#         self.top_k = top_k
#         self.alpha = alpha
#         self.bm25 = None
#         self.docs = []
#         self.initialized = False
#
#     def initialize_bm25(self, documents: List[Dict]):
#         """初始化BM25索引"""
#         self.docs = documents
#         # 对文档进行分词
#         tokenized_docs = [list(jieba.cut(doc["text"])) for doc in documents]
#
#         self.bm25 = BM25Okapi(tokenized_docs)
#         self.initialized = True
#
#     def run(self,
#             query: Optional[str] = None,
#             query_embedding: Optional[np.ndarray] = None,
#             documents: Optional[List[Dict]] = None,
#             **kwargs) -> Tuple[Dict, Optional[str]]:
#         """
#         执行混合检索
#
#         Args:
#             query: 查询文本
#             query_embedding: 查询文本的向量表示
#             documents: 首次运行时需要提供用于构建BM25索引的文档
#         """
#         if documents and not self.initialized:
#             self.initialize_bm25(documents)
#
#         if not self.initialized:
#             raise RuntimeError("检索器未初始化,请先提供文档构建索引")
#
#         # BM25检索
#         tokenized_query = list(jieba.cut(query))
#         bm25_scores = self.bm25.get_scores(tokenized_query)
#
#         # # 打印一些最高分的文档内容
#         # top_indices = np.argsort(bm25_scores)[-5:][::-1]
#         # for idx in top_indices:
#         #     print(f"文档 {idx}: {self.docs[idx]['text'][:100]}... 分数: {bm25_scores[idx]}")
#
#         # 向量检索
#         vector_results = self.vector_store.run(
#             query=query,
#             query_embedding=query_embedding
#         )[0]
#
#         # 获取向量检索分数
#         vector_scores = np.zeros(len(self.docs))
#         for result in vector_results["results"]:
#             doc_idx = self._find_doc_index(result["text"])
#             if doc_idx != -1:
#                 vector_scores[doc_idx] = result["score"]
#
#         # 归一化分数
#         bm25_scores = self._normalize_scores(bm25_scores)
#         vector_scores = self._normalize_scores(vector_scores)
#
#         # 融合分数
#         # final_scores = self.alpha * bm25_scores + (1 - self.alpha) * vector_scores
#         final_scores = bm25_scores
#         # 获取top-k结果
#         top_k_indices = np.argsort(final_scores)[-self.top_k:][::-1]
#         results = []
#
#         for idx in top_k_indices:
#             doc = self.docs[idx]
#             results.append({
#                 "text": doc["text"],
#                 "score": float(final_scores[idx]),
#                 "source": doc.get("source", ""),
#                 "metadata": doc.get("metadata", {}),
#             })
#
#         output = {
#             "query": query,
#             "results": results,
#             "total_docs": len(self.docs)
#         }
#
#         self.save_output(output)
#         return output, None
#
#     def _find_doc_index(self, text: str) -> int:
#         """查找文档在列表中的索引"""
#         for i, doc in enumerate(self.docs):
#             if doc["text"] == text:
#                 return i
#         return -1
#
#     def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
#         """归一化分数到[0,1]区间"""
#         if np.all(scores == 0):
#             return scores
#         return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

# nodes/retriever/hybrid_retriever.py
import numpy as np
from typing import Dict, List, Optional
import logging
from rank_bm25 import BM25Okapi
from nodes.base import BaseComponent
from storage.base import BaseStorage
from pipeline_data import PipelineData
import jieba

logger = logging.getLogger(__name__)


class HybridRetrieverNode(BaseComponent):
    """混合检索节点,结合向量检索和BM25检索"""

    def __init__(self,
                 vector_store_node,
                 top_k: int = 3,
                 alpha: float = 0.5,  # BM25和向量检索的权重因子
                 storage: Optional[BaseStorage] = None):
        """
        初始化混合检索节点

        Args:
            vector_store_node: 向量存储节点实例
            top_k: 返回的最相关文档数量
            alpha: BM25分数的权重(0-1),剩余权重分配给向量检索分数
            storage: 可选的存储实例
        """
        super().__init__(storage=storage)
        self.vector_store = vector_store_node
        self.top_k = top_k
        self.alpha = alpha
        self.bm25 = None
        self.docs = []
        self.initialized = False

    def initialize_bm25(self, documents: List[Dict]):
        """初始化BM25索引"""
        self.docs = documents
        # 对文档进行分词
        tokenized_docs = [list(jieba.cut(doc["text"])) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.initialized = True

    def run(self, data: PipelineData, **kwargs) -> PipelineData:
        """
        执行混合检索
        输入: data.content 应该是查询向量
        输出: PipelineData 包含混合检索结果
        """
        query = data.get_metadata('query', '')
        query_embedding = data.content

        if not self.initialized:
            raise RuntimeError("检索器未初始化,请先提供文档构建索引")

        # BM25检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # 向量检索 - 直接调用向量存储节点
        vector_data = PipelineData(
            content=query_embedding,
            metadata={'query': query, 'node_type': 'query_embedding'},
            source='HybridRetriever'
        )
        vector_results = self.vector_store.run(vector_data)

        # 获取向量检索分数
        vector_scores = np.zeros(len(self.docs))
        for result in vector_results.content:
            doc_idx = self._find_doc_index(result["text"])
            if doc_idx != -1:
                vector_scores[doc_idx] = result["score"]

        # 归一化分数
        bm25_scores = self._normalize_scores(bm25_scores)
        vector_scores = self._normalize_scores(vector_scores)

        # 融合分数 - 这里主要使用BM25分数
        final_scores = bm25_scores  # 或者 self.alpha * bm25_scores + (1 - self.alpha) * vector_scores

        # 获取top-k结果
        top_k_indices = np.argsort(final_scores)[-self.top_k:][::-1]
        results = []

        for idx in top_k_indices:
            doc = self.docs[idx]
            results.append({
                "text": doc["text"],
                "score": float(final_scores[idx]),
                "source": doc.get("source", ""),
                "metadata": doc.get("metadata", {}),
                "chunk_id": doc.get("chunk_id", f"chunk_{idx}")
            })

        result = PipelineData(
            content=results,
            metadata={
                'query': query,
                'total_docs': len(self.docs),
                'node_type': 'hybrid_retrieval',
                'alpha': self.alpha
            },
            source='HybridRetriever'
        )

        self.save_output(result)
        return result

    def _find_doc_index(self, text: str) -> int:
        """查找文档在列表中的索引"""
        for i, doc in enumerate(self.docs):
            if doc["text"] == text:
                return i
        return -1

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """归一化分数到[0,1]区间"""
        if np.all(scores == 0):
            return scores
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))