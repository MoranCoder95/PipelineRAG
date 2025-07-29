# nodes/document/text_splitter.py
from typing import List, Dict, Optional
import logging
from nodes.base import BaseNode
from storage.base import BaseStorage
from pipeline_data import PipelineData

logger = logging.getLogger(__name__)


class TextSplitterNode(BaseNode):
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separator: str = "\n",
                 storage: Optional[BaseStorage] = None):
        super().__init__(storage=storage)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def run(self, data: PipelineData, **kwargs) -> PipelineData:
        """
        分割文本
        输入: data.content 应该是文档列表
        输出: PipelineData 包含分割后的文本块
        """
        documents = data.content
        if not documents:
            logger.error("没有收到有效的文档数据")
            raise ValueError("没有收到有效的文档数据")

        logger.info(f"开始处理文档块，输入统计:")
        logger.info(f"- 总文档数: {len(documents)}")

        all_chunks = []
        processed_docs = 0
        total_chunks = 0

        # 处理每个文档块
        for doc in documents:
            try:
                # 提取文档信息
                content = doc.get('content', '')
                source = doc.get('source', 'unknown')
                metadata = doc.get('metadata', {})

                # 分割文本
                text_chunks = self._split_text(content)

                # 为每个块添加元数据
                for i, chunk_text in enumerate(text_chunks):
                    chunk = {
                        'text': chunk_text,
                        'source': source,
                        'metadata': metadata,
                        'chunk_id': f"{source}_{i}",
                        'chunk_index': i,
                        'chunk_size': len(chunk_text),
                        'original_file': source
                    }
                    all_chunks.append(chunk)

                total_chunks += len(text_chunks)
                processed_docs += 1
                logger.debug(f"成功处理文档: {source}, 生成 {len(text_chunks)} 个文本块")

            except Exception as e:
                logger.error(f"处理文档时出错 {source}: {str(e)}")
                continue

        # 返回统一格式
        result = PipelineData(
            content=all_chunks,
            metadata={
                'total_chunks': total_chunks,
                'processed_documents': processed_docs,
                'original_docs': len(documents),
                'node_type': 'text_splitter',
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'chunk_stats': {
                    'avg_chunk_size': sum(len(c['text']) for c in all_chunks) / len(all_chunks) if all_chunks else 0,
                    'max_chunk_size': max(len(c['text']) for c in all_chunks) if all_chunks else 0,
                    'min_chunk_size': min(len(c['text']) for c in all_chunks) if all_chunks else 0
                },
                'processing_stats': data.metadata
            },
            source='TextSplitter'
        )

        logger.info(f"文本分割完成:")
        logger.info(f"- 处理文档数: {processed_docs}")
        logger.info(f"- 生成文本块: {total_chunks}")
        logger.info(f"- 平均块大小: {result.metadata['chunk_stats']['avg_chunk_size']:.2f} 字符")

        self.save_output(result)
        return result

    def _split_text(self, text: str) -> List[str]:
        """简化的文本分割"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks

    def _split_text_advanced(self, text: str) -> List[str]:
        """高级文本分块的具体实现"""
        if len(text) <= self.chunk_size:
            return [text]

        segments = text.split(self.separator)
        current_chunk = []
        current_length = 0
        chunks = []

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            segment_length = len(segment)

            if segment_length > self.chunk_size:
                if current_chunk:
                    chunks.append(self.separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                sub_chunks = self._split_large_segment(segment)
                chunks.extend(sub_chunks)
                continue

            if current_length + segment_length > self.chunk_size:
                chunks.append(self.separator.join(current_chunk))
                current_chunk = [segment]
                current_length = segment_length
            else:
                current_chunk.append(segment)
                current_length += segment_length

        if current_chunk:
            chunks.append(self.separator.join(current_chunk))

        # 添加重叠
        final_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_start = max(0, len(prev_chunk) - self.chunk_overlap)
                chunk = prev_chunk[overlap_start:] + self.separator + chunk
            final_chunks.append(chunk)

        return final_chunks

    def _split_large_segment(self, segment: str) -> List[str]:
        """处理超长段落"""
        import re
        sentences = re.split('([。！？.!?])', segment)
        current_chunk = []
        current_length = 0
        chunks = []

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            sentence_length = len(sentence)

            if sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                for j in range(0, len(sentence), self.chunk_size):
                    chunks.append(sentence[j:j + self.chunk_size])
                continue

            if current_length + sentence_length > self.chunk_size:
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(''.join(current_chunk))

        return chunks
