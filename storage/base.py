# storage/base.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List


class BaseStorage(ABC):
    """存储基类"""

    subclasses: dict = {}

    def __init_subclass__(cls, **kwargs):
        """注册子类"""
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def get_subclass(cls, storage_type: str):
        """获取对应的子类

        Args:
            storage_type: 存储类型名称
        """
        if storage_type not in cls.subclasses:
            raise Exception(f"找不到名为 '{storage_type}' 的存储类型。")
        return cls.subclasses[storage_type]

    @abstractmethod
    def save(self, data: Any):
        """保存数据"""
        pass

    @abstractmethod
    def load(self) -> Any:
        """加载数据"""
        pass

    @abstractmethod
    def close(self):
        """关闭存储"""
        pass