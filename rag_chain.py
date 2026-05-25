"""
RAG问答链模块
整合检索器和Ollama大模型构建完整的RAG问答系统
AI辅助生成
"""
from typing import Optional, List, Dict
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain.schema import Document
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
        self.chain: Optional[ConversationalRetrievalChain] = None
        self.vector_manager = VectorStoreManager()

    def initialize_chain(self) -> bool:
        """
        初始化问答链

        Returns:
            是否成功初始化
        """
        try:
            retriever = self.vector_manager.get_retriever()
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": self._create_prompt()}
            )
            print("RAG问答链初始化成功!")
            return True
        except Exception as e:
            print(f"RAG问答链初始化失败: {str(e)}")
            return False

    def _create_prompt(self):
        """创建自定义提示词"""
        from langchain.prompts import PromptTemplate

        return PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )

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

    def ask(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict:
        """
        提问

        Args:
            question: 用户问题
            chat_history: 对话历史

        Returns:
            包含答案和相关文档的字典
        """
        if self.chain is None:
            if not self.initialize_chain():
                return {
                    "answer": "RAG问答链未初始化，请先构建知识库",
                    "source_documents": []
                }

        if chat_history is None:
            chat_history = []

        try:
            result = self.chain({
                "question": question,
                "chat_history": chat_history
            })

            return {
                "answer": result.get("answer", ""),
                "source_documents": result.get("source_documents", [])
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

        print("\n初始化问答链...")
        rag_chain.initialize_chain()

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