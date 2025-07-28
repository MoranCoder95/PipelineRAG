# example_usage.py - 使用示例
import os
from pathlib import Path
from pipeline_data import PipelineData
from nodes.document.pdf_processor import PDFProcessorNode
from nodes.document.text_splitter import TextSplitterNode
from nodes.embedding.text_embedding import TextEmbeddingNode
from nodes.vector.vector_store_node import VectorStoreNode
from nodes.llm.openai import GPT3Node
from nodes.prompt.prompt_node import PromptNode
from nodes.retriever.hybrid_retriever import HybridRetrieverNode
from storage.excel_storage import ExcelStorage
from pipelines.base import Pipeline


def get_all_pdf_files(root_dir: str) -> list:
    """递归获取所有PDF文件"""
    pdf_files = []
    root_path = Path(root_dir)

    for file_path in root_path.rglob("*.pdf"):
        try:
            if file_path.is_file():
                pdf_files.append(str(file_path.resolve()))
        except (PermissionError, OSError) as e:
            print(f"无法访问文件 {file_path}: {str(e)}")
            continue

    return pdf_files


def create_indexing_pipeline():
    """创建文档索引pipeline"""
    pipeline = Pipeline()

    # 创建组件
    pdf_processor = PDFProcessorNode(
        storage=ExcelStorage("data/pdf_processing.xlsx")
    )

    text_splitter = TextSplitterNode(
        chunk_size=512,
        chunk_overlap=50,
        storage=ExcelStorage("data/text_chunks.xlsx")
    )

    embedding_node = TextEmbeddingNode(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        storage=ExcelStorage("data/embeddings.xlsx")
    )

    vector_store = VectorStoreNode(
        index_file_path="data/faiss_index.bin",
        mapping_file_path="data/text_mapping.pkl",
        dimension=384
    )

    # 添加节点到pipeline
    pipeline.add_node(component=pdf_processor, name="PDFProcessor", inputs=["File"])
    pipeline.add_node(component=text_splitter, name="TextSplitter", inputs=["PDFProcessor"])
    pipeline.add_node(component=embedding_node, name="TextEmbedding", inputs=["TextSplitter"])
    pipeline.add_node(component=vector_store, name="VectorStore", inputs=["TextEmbedding"])

    return pipeline


def create_query_pipeline(documents=None):
    """创建查询pipeline"""
    pipeline = Pipeline()

    embedding_node = TextEmbeddingNode(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    vector_store = VectorStoreNode(
        index_file_path="data/faiss_index.bin",
        mapping_file_path="data/text_mapping.pkl",
        dimension=384
    )

    # 创建混合检索器
    hybrid_retriever = HybridRetrieverNode(
        vector_store_node=vector_store,
        top_k=3,
        alpha=0.5
    )

    # 如果有文档，初始化BM25索引
    if documents:
        hybrid_retriever.initialize_bm25(documents)

    prompt_node = PromptNode(
        storage=ExcelStorage("data/prompts.xlsx")
    )

    gpt_node = GPT3Node(
        api_key="Bearer your-api-key-here",
        storage=ExcelStorage("data/gpt_responses.xlsx")
    )

    # 添加节点到pipeline
    pipeline.add_node(component=embedding_node, name="TextEmbedding", inputs=["Query"])
    pipeline.add_node(component=hybrid_retriever, name="HybridRetrieval", inputs=["TextEmbedding"])
    pipeline.add_node(component=prompt_node, name="PromptBuilder", inputs=["HybridRetrieval"])
    pipeline.add_node(component=gpt_node, name="GPTNode", inputs=["PromptBuilder"])

    return pipeline


def main():
    """主函数"""
    # 设置基础存储目录
    os.makedirs("data", exist_ok=True)

    # 获取所有PDF文件
    pdf_dir = "doc"  # 你的文档根目录
    pdf_files = get_all_pdf_files(pdf_dir)

    if not pdf_files:
        print(f"在 {pdf_dir} 目录下未找到PDF文件")
        return

    print(f"找到 {len(pdf_files)} 个PDF文件:")
    for file in pdf_files:
        print(f"- {file}")

    # 确认是否继续
    response = input("\n是否开始处理这些文件? (y/n): ")
    if response.lower() != 'y':
        print("操作已取消")
        return

    # 第一步：处理文档并建立索引
    print("\n开始处理文档...")
    indexing_pipeline = create_indexing_pipeline()

    # 使用PipelineData包装输入
    initial_data = PipelineData(
        content=pdf_files,
        metadata={'node_type': 'file_paths'},
        source='Main'
    )

    # 运行索引pipeline
    index_results = indexing_pipeline.run(file_paths=pdf_files)

    # 从结果中提取文档块
    # 注意：这里需要根据实际的pipeline输出结构来调整
    documents = []
    if 'chunks' in index_results:
        documents = index_results['chunks']

    print("文档处理完成！\n")

    # 第二步：设置查询pipeline
    query_pipeline = create_query_pipeline(documents)

    print("进入问答模式，您可以开始提问了！")
    print("提示：您可以询问关于任何已处理文档中的内容")
    print("输入 'q' 退出程序\n")

    # 第三步：查询循环
    while True:
        query = input("\n请输入您的问题: ")
        if query.lower() == 'q':
            break

        try:
            result = query_pipeline.run(query=query)
            print(f"\nAI回答: {result['response']}")
        except Exception as e:
            print(f"\n处理问题时出错: {str(e)}")
            print("请重试或输入新的问题")


def simple_usage_example():
    """简单使用示例"""
    print("=== 简单的RAG Pipeline使用示例 ===\n")

    # 创建一个简单的pipeline
    pipeline = Pipeline()

    # 添加组件
    pipeline.add_node(PDFProcessorNode(), "pdf_processor", ["File"])
    pipeline.add_node(TextSplitterNode(chunk_size=500), "text_splitter", ["pdf_processor"])
    pipeline.add_node(TextEmbeddingNode(), "text_embedding", ["text_splitter"])
    pipeline.add_node(VectorStoreNode("data/index.bin", "data/mapping.pkl"), "vector_store", ["text_embedding"])

    # 运行pipeline处理文档
    pdf_files = ["doc1.pdf", "doc2.pdf"]  # 替换为实际的PDF文件路径

    try:
        result = pipeline.run(file_paths=pdf_files)
        print(f"处理完成，状态: {result}")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")


if __name__ == "__main__":
    try:
        # 运行完整示例
        main()

        # 或者运行简单示例
        # simple_usage_example()

    except KeyboardInterrupt:
        print("\n\n程序已被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
    finally:
        print("\n程序已退出")