"""
向量存储和检索模块
使用Chroma向量数据库存储文档嵌入
AI辅助生成
"""
import os
import shutil
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from config import CHROMA_DB_PATH, EMBEDDING_MODEL, TOP_K


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(
        self,
        persist_directory: str = CHROMA_DB_PATH,
        embedding_model: str = EMBEDDING_MODEL
    ):
        """
        初始化向量存储管理器

        Args:
            persist_directory: 向量数据库持久化路径
            embedding_model: 嵌入模型名称
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self._vectorstore: Optional[Chroma] = None

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        从文档创建向量存储

        Args:
            documents: 文档列表

        Returns:
            Chroma向量存储对象
        """
        if not documents:
            raise ValueError("文档列表为空，无法创建向量存储")

        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"已清除旧的向量数据库: {self.persist_directory}")

        print(f"正在创建向量数据库...")
        print(f"  嵌入模型: {self.embedding_model}")
        print(f"  文档数量: {len(documents)}")

        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        self._vectorstore.persist()
        print(f"向量数据库已保存至: {self.persist_directory}")

        return self._vectorstore

    def get_vectorstore(self) -> Optional[Chroma]:
        """
        获取已加载的向量存储

        Returns:
            Chroma向量存储对象，如果未加载则返回None
        """
        if self._vectorstore is None:
            if os.path.exists(self.persist_directory):
                print(f"正在加载已有向量数据库: {self.persist_directory}")
                self._vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
        return self._vectorstore

    def get_retriever(self, top_k: int = TOP_K):
        """
        获取检索器

        Args:
            top_k: 返回的最相关文档数量

        Returns:
            检索器对象
        """
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            raise ValueError("向量存储未初始化，请先创建或加载向量数据库")

        return vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )

    def similarity_search(self, query: str, top_k: int = TOP_K) -> List[Document]:
        """
        相似性搜索

        Args:
            query: 查询文本
            top_k: 返回的最相关文档数量

        Returns:
            最相关的文档列表
        """
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            raise ValueError("向量存储未初始化，请先创建或加载向量数据库")

        return vectorstore.similarity_search(query, k=top_k)

    def get_chunk_count(self) -> int:
        """获取当前知识库中的文本块数量"""
        vectorstore = self.get_vectorstore()
        if vectorstore is None:
            return 0
        return vectorstore._collection.count()

    def clear_vectorstore(self):
        """清空向量数据库"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"已清空向量数据库: {self.persist_directory}")
        self._vectorstore = None


if __name__ == "__main__":
    from document_loader import DocumentLoader
    from text_splitter import TextSplitter
    from config import DOCUMENTS_PATH

    print("=" * 50)
    print("向量存储和检索测试")
    print("=" * 50)

    loader = DocumentLoader(DOCUMENTS_PATH)
    documents = loader.load_all_documents()

    if documents:
        splitter = TextSplitter()
        split_docs = splitter.split_documents(documents)

        manager = VectorStoreManager()

        manager.create_vectorstore(split_docs)

        query = "什么是自然语言处理？"
        print(f"\n执行检索测试，查询: {query}")

        results = manager.similarity_search(query)

        print(f"\n检索结果 (返回 {len(results)} 个相关文档块):")
        for i, doc in enumerate(results):
            print(f"\n--- 结果 {i+1} ---")
            print(f"  内容: {doc.page_content[:150]}...")
    else:
        print("\n未找到文档，请先将文档放入 documents 文件夹")