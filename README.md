# PipelineRAG - 模块化RAG系统

基于PipelineData统一数据结构的模块化检索增强生成(RAG)系统。

## 📁 项目结构

```
PipelineRAG/
├── pipeline_data.py              # 统一数据结构
├── example_usage.py             # 使用示例
├── nodes/                       # 处理节点
│   ├── __init__.py
│   ├── base.py                  # 基础组件类
│   ├── document/                # 文档处理
│   │   ├── __init__.py
│   │   ├── pdf_processor.py     # PDF处理节点
│   │   └── text_splitter.py     # 文本分割节点
│   ├── embedding/               # 文本嵌入
│   │   ├── __init__.py
│   │   └── text_embedding.py    # 文本向量化节点
│   ├── llm/                     # 大语言模型
│   │   ├── __init__.py
│   │   └── openai.py           # OpenAI接口节点
│   ├── prompt/                  # 提示构建
│   │   ├── __init__.py
│   │   └── prompt_node.py      # 提示构建节点
│   ├── retriever/               # 检索器
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py   # BM25检索节点
│   │   ├── hybrid_retriever.py # 混合检索节点
│   │   └── vector_retriever.py # 向量检索节点
│   └── vector/                  # 向量存储
│       ├── __init__.py
│       └── vector_store_node.py # 向量存储节点
├── pipelines/                   # Pipeline管理
│   ├── __init__.py
│   ├── base.py                 # Pipeline基础类
│   └── config.py               # 配置处理
├── storage/                     # 存储接口
│   ├── __init__.py
│   ├── base.py                 # 存储基类
│   └── excel_storage.py        # Excel存储实现
└── utils/                       # 工具函数
```

## 🏗️ 核心架构

### PipelineData 统一数据结构

```python
@dataclass
class PipelineData:
    content: Any                    # 主要数据内容
    metadata: Dict[str, Any]        # 元数据信息
    source: Optional[str]           # 数据来源
```

### 组件化设计

- **BaseNode**: 所有处理节点的基类
- **Pipeline**: 管道编排和执行引擎
- **Storage**: 可插拔的存储后端

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install sentence-transformers faiss-cpu PyMuPDF openpyxl rank-bm25 jieba networkx
```


### 2. 基本使用示例

```python
from pipeline_data import PipelineData
from pipelines.base import Pipeline
from nodes.document.pdf_processor import PDFProcessorNode
from nodes.document.text_splitter import TextSplitterNode
from nodes.embedding.text_embedding import TextEmbeddingNode
from nodes.vector.vector_store_node import VectorStoreNode

# 创建索引pipeline
pipeline = Pipeline()
pipeline.add_node(PDFProcessorNode(), "pdf_processor", ["File"])
pipeline.add_node(TextSplitterNode(chunk_size=500), "text_splitter", ["pdf_processor"])
pipeline.add_node(TextEmbeddingNode(), "text_embedding", ["text_splitter"])
pipeline.add_node(VectorStoreNode("data/index.bin", "data/mapping.pkl"), "vector_store", ["text_embedding"])

# 处理文档
pdf_files = ["doc1.pdf", "doc2.pdf"]
result = pipeline.run(file_paths=pdf_files)
```

## 🔧 主要特性

### 1. 统一数据格式
- 所有组件间使用PipelineData进行数据传递
- 标准化的元数据管理
- 便于调试和监控

### 2. 模块化设计
- 每个功能封装为独立节点
- 可自由组合和替换组件
- 支持复杂的处理流程

### 3. 多种检索方式
- **向量检索**: 基于语义相似度
- **BM25检索**: 基于关键词匹配
- **混合检索**: 结合两种方式的优点

### 4. 灵活的存储后端
- Excel存储：便于查看和分析
- 可扩展其他存储方式

### 5. 完整的RAG流程
- 文档解析 → 文本分割 → 向量化 → 存储 → 检索 → 生成回答

## 📝 核心组件说明

### 文档处理
- **PDFProcessorNode**: 提取PDF文本内容，保留文档结构
- **TextSplitterNode**: 智能文本分割，支持重叠和上下文保持

### 向量化
- **TextEmbeddingNode**: 使用sentence-transformers生成文本向量
- 支持中英文多语言模型

### 检索
- **VectorStoreNode**: FAISS向量存储和检索
- **HybridRetrieverNode**: 混合检索，平衡语义和关键词匹配
- **BM25RetrieverNode**: 传统关键词检索

### 生成
- **PromptNode**: 智能提示构建，格式化检索结果
- **GPT3Node**: OpenAI API接口，支持多种模型

## 🎯 使用场景

1. **企业知识库**: 处理内部文档，提供智能问答
2. **学术研究**: 分析大量论文和资料
3. **客服系统**: 基于产品文档的自动回答
4. **教育应用**: 构建课程内容问答系统


## 🔍 监控和调试

系统提供完整的执行链路跟踪：

1. **数据流跟踪**: 每个节点的输入输出
2. **元数据记录**: 处理统计和状态信息
3. **存储记录**: 可选的Excel存储，便于分析
4. **错误处理**: 详细的错误信息和恢复机制

## 🚧 扩展开发

### 添加新的处理节点

```python
from nodes.base import BaseNode
from pipeline_data import PipelineData

class CustomNode(BaseNode):
    def run(self, data: PipelineData, **kwargs) -> PipelineData:
        # 处理逻辑
        processed_content = self.process(data.content)
        
        # 返回新的PipelineData
        return PipelineData(
            content=processed_content,
            metadata={'node_type': 'custom_processing'},
            source='CustomNode'
        )
```

### 添加新的存储后端

```python
from storage.base import BaseStorage

class CustomStorage(BaseStorage):
    def save(self, data):
        # 实现保存逻辑
        pass
    
    def load(self):
        # 实现加载逻辑
        pass
    
    def close(self):
        # 实现清理逻辑
        pass
```

## 📊 性能优化

1. **向量索引**: 使用FAISS进行高效相似度搜索
2. **文档缓存**: 避免重复处理已索引文档
3. **批处理**: 支持批量向量化提升效率
4. **增量更新**: 支持增量添加新文档

