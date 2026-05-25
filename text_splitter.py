"""
文本分割模块
使用RecursiveCharacterTextSplitter对文档进行分块处理
AI辅助生成
"""
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP


class TextSplitter:
    """文本分割器类"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        初始化文本分割器

        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻文本块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档列表分割成更小的文本块

        Args:
            documents: 原始文档列表

        Returns:
            分割后的文档列表
        """
        if not documents:
            return []

        print(f"正在分割 {len(documents)} 个文档块...")
        print(f"  chunk_size: {self.chunk_size}")
        print(f"  chunk_overlap: {self.chunk_overlap}")

        split_docs = self.splitter.split_documents(documents)

        print(f"分割完成，获得 {len(split_docs)} 个文本块")
        return split_docs

    def split_texts(self, texts: List[str]) -> List[str]:
        """
        将文本列表分割成更小的文本块

        Args:
            texts: 原始文本列表

        Returns:
            分割后的文本块列表
        """
        if not texts:
            return []

        return self.splitter.split_text("\n\n".join(texts))


if __name__ == "__main__":
    from document_loader import DocumentLoader
    from config import DOCUMENTS_PATH

    print("=" * 50)
    print("文本分割测试")
    print("=" * 50)

    loader = DocumentLoader(DOCUMENTS_PATH)
    documents = loader.load_all_documents()

    if documents:
        splitter = TextSplitter()
        split_docs = splitter.split_documents(documents)

        print(f"\n分割结果示例:")
        if split_docs:
            print(f"  总文本块数: {len(split_docs)}")
            print(f"  第一个文本块长度: {len(split_docs[0].page_content)}")
            print(f"  第一个文本块预览: {split_docs[0].page_content[:150]}...")