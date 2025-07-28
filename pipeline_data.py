# # 1. 统一的数据结构
# from dataclasses import dataclass, field
# from typing import Any, Dict, Optional, List, Tuple
# import logging
#
# logger = logging.getLogger(__name__)
#
#
# @dataclass
# class PipelineData:
#     """统一的Pipeline数据格式"""
#     content: Any  # 主要数据内容
#     metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
#     source: Optional[str] = None  # 数据来源
#
#     def to_dict(self) -> Dict:
#         """转换为字典格式"""
#         return {
#             'content': self.content,
#             'metadata': self.metadata,
#             'source': self.source
#         }

# pipeline_data.py
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineData:
    """统一的Pipeline数据格式"""
    content: Any  # 主要数据内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    source: Optional[str] = None  # 数据来源

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'content': self.content,
            'metadata': self.metadata,
            'source': self.source
        }

    def add_metadata(self, key: str, value: Any):
        """添加元数据"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default=None):
        """获取元数据"""
        return self.metadata.get(key, default)

    def update_metadata(self, metadata: Dict[str, Any]):
        """更新元数据"""
        self.metadata.update(metadata)