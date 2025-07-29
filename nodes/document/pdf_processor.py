# nodes/document/pdf_processor.py
import os
import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import Dict, List, Optional
from nodes.base import BaseNode
from storage.base import BaseStorage
from pipeline_data import PipelineData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessorNode(BaseNode):
    """增强的PDF处理节点"""

    def __init__(self,
                 storage: Optional[BaseStorage] = None,
                 min_line_length: int = 10,
                 encoding_check: bool = True,
                 extract_metadata: bool = True,
                 preserve_sections: bool = True):
        """
        初始化PDF处理节点

        Args:
            storage: 可选的存储实例
            min_line_length: 最小行长度，小于此长度的行会被过滤
            encoding_check: 是否检查和处理编码问题
            extract_metadata: 是否提取详细元数据
            preserve_sections: 是否保留文档章节结构
        """
        super().__init__(storage=storage)
        self.min_line_length = min_line_length
        self.encoding_check = encoding_check
        self.extract_metadata = extract_metadata
        self.preserve_sections = preserve_sections

    def run(self, data: PipelineData, **kwargs) -> PipelineData:
        """
        处理PDF文件
        输入: data.content 应该是文件路径列表
        输出: PipelineData 包含提取的文档内容
        """
        file_paths = data.content
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        processed_docs = []
        processed_files = []
        failed_files = []

        total_files = len(file_paths)
        for index, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"处理文件 [{index}/{total_files}]: {Path(file_path).name}")

                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"找不到文件: {file_path}")

                if not file_path.lower().endswith('.pdf'):
                    logger.warning(f"跳过非PDF文件: {file_path}")
                    continue

                # 处理PDF文件
                text_content, metadata = self._process_pdf(file_path)
                if text_content:
                    processed_docs.append({
                        'content': text_content,
                        'source': str(Path(file_path).name),
                        'metadata': metadata,
                        'type': 'document'
                    })
                    processed_files.append(file_path)
                    logger.info(f"成功处理文件: {Path(file_path).name}")

            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {str(e)}")
                failed_files.append((file_path, str(e)))
                continue

        # 返回统一格式
        result = PipelineData(
            content=processed_docs,
            metadata={
                'total_files': len(file_paths),
                'processed_files': len(processed_docs),
                'successful_files': processed_files,
                'failed_files': failed_files,
                'failed_count': len(failed_files),
                'node_type': 'pdf_processor'
            },
            source='PDFProcessor'
        )

        logger.info(f"处理完成:")
        logger.info(f"- 总文件数: {total_files}")
        logger.info(f"- 成功处理: {len(processed_files)}")
        logger.info(f"- 处理失败: {len(failed_files)}")

        if failed_files:
            logger.info("\n失败的文件:")
            for file_path, error in failed_files:
                logger.info(f"- {Path(file_path).name}: {error}")

        self.save_output(result)
        return result

    def _process_pdf(self, file_path: str) -> tuple[str, Dict]:
        """
        处理单个PDF文件

        Args:
            file_path: PDF文件路径

        Returns:
            tuple[str, Dict]: (提取的文本内容, 元数据)
        """
        doc = fitz.open(file_path)
        text_parts = []

        try:
            # 提取元数据
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'page_count': len(doc),
                'file_name': Path(file_path).name
            }

            # 提取文档结构
            if self.extract_metadata:
                doc_structure = self._extract_document_structure(doc)
                metadata.update(doc_structure)

            for page_num, page in enumerate(doc, 1):
                try:
                    if self.preserve_sections and self.extract_metadata:
                        processed_text = self._process_page_with_context(
                            page, page_num, metadata
                        )
                    else:
                        # 简单处理
                        text = page.get_text()
                        if self.encoding_check:
                            text = self._handle_encoding(text)

                        lines = [line.strip() for line in text.split('\n')]
                        lines = [line for line in lines if len(line) >= self.min_line_length]

                        if lines:
                            processed_text = f"[第{page_num}页]\n" + '\n'.join(lines)
                        else:
                            continue

                    text_parts.append(processed_text)

                except Exception as e:
                    logger.warning(f"处理第 {page_num} 页时出错: {str(e)}")
                    continue

        finally:
            doc.close()

        return '\n\n'.join(text_parts), metadata

    def _extract_document_structure(self, doc) -> Dict:
        """提取文档结构信息"""
        structure = {
            'sections': []
        }

        # 尝试提取目录结构
        toc = doc.get_toc()
        if toc:
            for level, title, _ in toc:
                structure['sections'].append({
                    'level': level,
                    'title': title
                })

        return structure

    def _process_page_with_context(self, page, page_num: int, doc_structure: Dict) -> str:
        """处理单个页面，保留上下文信息"""
        text = page.get_text()

        # 处理编码问题
        if self.encoding_check:
            text = self._handle_encoding(text)

        # 处理文本行
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if len(line) >= self.min_line_length]

        # 构建上下文信息
        context = []

        # 添加文档基本信息
        if doc_structure.get('title'):
            context.append(f"文档标题: {doc_structure['title']}")

        # 查找当前页面所属章节
        current_section = None
        if self.preserve_sections and doc_structure.get('sections'):
            for section in doc_structure['sections']:
                # 这里可以根据具体需求优化章节匹配逻辑
                if any(section['title'] in line for line in lines):
                    current_section = section
                    break

        if current_section:
            context.append(f"当前章节: {current_section['title']}")

        # 添加页码信息
        context.append(f"页码: {page_num}")

        # 组合上下文和内容
        if context:
            return "\n".join([
                "---文档上下文---",
                "\n".join(context),
                "---页面内容---",
                "\n".join(lines)
            ])
        else:
            return f"[第{page_num}页]\n" + '\n'.join(lines)

    def _handle_encoding(self, text: str) -> str:
        """处理文本编码问题"""
        try:
            if not isinstance(text, str):
                text = text.decode('utf-8', errors='ignore')
            text = text.replace('\x00', '')  # 删除空字符
            text = text.replace('\ufeff', '')  # 删除BOM
            return text.strip()
        except Exception as e:
            logger.warning(f"处理编码时出错: {str(e)}")
            return text.strip()