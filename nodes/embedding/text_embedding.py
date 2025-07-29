# nodes/embedding/text_embedding.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import logging
from nodes.base import BaseNode
from storage.base import BaseStorage
from pipeline_data import PipelineData

logger = logging.getLogger(__name__)


class TextEmbeddingNode(BaseNode):
    def __init__(self,
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 storage: Optional[BaseStorage] = None):
        """
        初始化文本嵌入节点

        Args:
            model_name: Sentence Transformer模型名称
            storage: 可选的存储实例
        """
        super().__init__(storage=storage)
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """延迟加载模型"""
        self.model = SentenceTransformer(self.model_name)

    def run(self, data: PipelineData, **kwargs) -> PipelineData:
        """
        生成文本嵌入向量
        支持两种模式:
        1. 批量模式: data.content 是文档块列表
        2. 查询模式: data.content 是查询字符串
        """
        content = data.content

        if isinstance(content, str):
            # 查询模式
            logger.info("进入查询模式")
            embedding = self.model.encode([content])[0]
            result = PipelineData(
                content=embedding,
                metadata={
                    'query': content,
                    'model_name': self.model_name,
                    'dimension': len(embedding),
                    'node_type': 'query_embedding'
                },
                source='TextEmbedding'
            )
        else:
            # 批量模式
            logger.info(f"进入索引模式，处理 {len(content)} 个文本块")

            texts = [chunk['text'] for chunk in content]

            # 生成向量
            embeddings = self.model.encode(texts, show_progress_bar=True)

            # 为每个块添加向量
            enhanced_chunks = []
            for chunk, embedding in zip(content, embeddings):
                enhanced_chunk = chunk.copy()
                enhanced_chunk['embedding'] = embedding
                enhanced_chunk['embedding_dimension'] = len(embedding)
                enhanced_chunks.append(enhanced_chunk)

            result = PipelineData(
                content=enhanced_chunks,
                metadata={
                    'total_vectors': len(enhanced_chunks),
                    'model_name': self.model_name,
                    'dimension': len(embeddings[0]) if len(enhanced_chunks) > 0 else None,
                    'node_type': 'document_embedding',
                    'original_stats': data.metadata
                },
                source='TextEmbedding'
            )

            logger.info(f"向量生成完成:")
            logger.info(f"- 总向量数: {len(enhanced_chunks)}")
            logger.info(f"- 向量维度: {result.metadata['dimension']}")

        self.save_output(result)
        return result