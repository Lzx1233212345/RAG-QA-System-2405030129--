"""
RAG问答链模块
整合检索器和Ollama大模型构建完整的RAG问答系统
简化版本 - 兼容最新LangChain
"""
from typing import Optional, List, Dict
import requests
import json
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from vector_store import VectorStoreManager
from config import OLLAMA_BASE_URL, DEFAULT_MODEL


SYSTEM_PROMPT = """你是一个专业的智能问答助手。你的职责是基于提供的参考文档回答用户的问题。

重要规则：
1. 只使用参考文档中提供的信息来回答问题
2. 如果参考文档中没有包含回答问题所需的相关信息，请明确回复"文档中未找到相关答案"
3. 回答时要引用相关的参考内容，但不要直接复制文档内容
4. 保持回答清晰、准确、有条理
5. 如果需要，可以在回答中总结和解释文档内容

参考文档信息：
{context}

用户问题：{question}

请基于上述参考文档内容回答用户问题。
"""


class RAGChain:
    """RAG问答链类"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL
    ):
        """
        初始化RAG问答链

        Args:
            model_name: Ollama模型名称
            base_url: Ollama服务地址
        """
        self.model_name = model_name
        self.base_url = base_url
        self.llm = Ollama(model=model_name, base_url=base_url)
        self.vector_manager = VectorStoreManager()

    def build_knowledge_base(self, documents: List[Document]) -> bool:
        """
        构建知识库

        Args:
            documents: 文档列表

        Returns:
            是否成功构建
        """
        try:
            self.vector_manager.create_vectorstore(documents)
            print("知识库构建完成!")
            return True
        except Exception as e:
            print(f"知识库构建失败: {str(e)}")
            return False

    def _call_ollama(self, prompt: str) -> str:
        """
        直接调用Ollama API

        Args:
            prompt: 提示词

        Returns:
            模型回复
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            return f"调用Ollama时发生错误: {str(e)}"

    def ask(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict:
        """
        提问

        Args:
            question: 用户问题
            chat_history: 对话历史

        Returns:
            包含答案和相关文档的字典
        """
        try:
            # 检索相关文档
            retriever = self.vector_manager.get_retriever()
            docs = retriever.invoke(question)
            
            # 构建上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 构建提示词
            full_prompt = SYSTEM_PROMPT.format(context=context, question=question)
            
            # 调用Ollama
            answer = self._call_ollama(full_prompt)
            
            return {
                "answer": answer,
                "source_documents": docs
            }
        except Exception as e:
            return {
                "answer": f"处理问题时发生错误: {str(e)}",
                "source_documents": []
            }

    def get_knowledge_base_status(self) -> Dict:
        """获取知识库状态"""
        count = self.vector_manager.get_chunk_count()
        return {
            "chunk_count": count,
            "has_documents": count > 0
        }


if __name__ == "__main__":
    from document_loader import DocumentLoader
    from text_splitter import TextSplitter
    from config import DOCUMENTS_PATH

    print("=" * 50)
    print("RAG问答链测试")
    print("=" * 50)

    loader = DocumentLoader(DOCUMENTS_PATH)
    documents = loader.load_all_documents()

    if documents:
        splitter = TextSplitter()
        split_docs = splitter.split_documents(documents)

        rag_chain = RAGChain()

        print("\n正在构建知识库...")
        rag_chain.build_knowledge_base(split_docs)

        test_questions = [
            "什么是自然语言处理？",
            "Transformer模型是如何工作的？",
            "什么是注意力机制？",
            "今天天气怎么样？"  # 无关问题
        ]

        print("\n" + "=" * 50)
        print("问答测试")
        print("=" * 50)

        for i, question in enumerate(test_questions):
            print(f"\n【问题 {i+1}】{question}")
            result = rag_chain.ask(question)
            print(f"【回答】{result['answer'][:200]}...")
            print(f"【参考文档数】{len(result['source_documents'])}")
    else:
        print("\n未找到文档，请先将文档放入 documents 文件夹")