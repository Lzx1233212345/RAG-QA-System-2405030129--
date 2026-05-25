"""
文档加载模块
支持PDF和DOCX格式的文档读取
AI辅助生成
"""
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document


class DocumentLoader:
    """文档加载器类"""

    def __init__(self, documents_path: str):
        """
        初始化文档加载器

        Args:
            documents_path: 文档所在文件夹路径
        """
        self.documents_path = documents_path

    def load_single_document(self, file_path: str) -> List[Document]:
        """
        加载单个文档

        Args:
            file_path: 文档文件路径

        Returns:
            文档对象列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_extension}")

        return loader.load()

    def load_all_documents(self) -> List[Document]:
        """
        加载文件夹中的所有文档

        Returns:
            所有文档对象列表
        """
        if not os.path.exists(self.documents_path):
            os.makedirs(self.documents_path)
            print(f"已创建文档目录: {self.documents_path}")
            return []

        all_documents = []
        supported_extensions = [".pdf", ".docx"]

        for root, _, files in os.walk(self.documents_path):
            for file in files:
                file_extension = os.path.splitext(file)[1].lower()
                if file_extension in supported_extensions:
                    file_path = os.path.join(root, file)
                    try:
                        docs = self.load_single_document(file_path)
                        all_documents.extend(docs)
                        print(f"已加载: {file} ({len(docs)} 页/段落)")
                    except Exception as e:
                        print(f"加载文件失败 {file}: {str(e)}")

        return all_documents

    def get_document_count(self) -> int:
        """获取文档数量"""
        if not os.path.exists(self.documents_path):
            return 0

        count = 0
        for root, _, files in os.walk(self.documents_path):
            for file in files:
                if file.lower().endswith((".pdf", ".docx")):
                    count += 1
        return count


if __name__ == "__main__":
    from config import DOCUMENTS_PATH

    print("=" * 50)
    print("文档加载测试")
    print("=" * 50)

    loader = DocumentLoader(DOCUMENTS_PATH)
    documents = loader.load_all_documents()

    print(f"\n总共加载了 {len(documents)} 个文档块")

    if documents:
        print(f"\n第一个文档块预览:")
        print(f"  来源: {documents[0].metadata}")
        print(f"  内容: {documents[0].page_content[:200]}...")