"""
RAG智能问答系统 - Streamlit Web应用
基于本地知识库的智能问答系统
AI辅助生成
"""
import streamlit as st
import os
import shutil
from datetime import datetime
from typing import List
from document_loader import DocumentLoader
from text_splitter import TextSplitter
from rag_chain import RAGChain
from config import DOCUMENTS_PATH, CHROMA_DB_PATH


def initialize_session_state():
    """初始化会话状态"""
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "knowledge_base_built" not in st.session_state:
        st.session_state.knowledge_base_built = False
    if "messages" not in st.session_state:
        st.session_state.messages = []


def save_uploaded_file(uploaded_file) -> str:
    """
    保存上传的文件到文档目录

    Args:
        uploaded_file: 上传的文件对象

    Returns:
        保存后的文件路径
    """
    os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    file_path = os.path.join(DOCUMENTS_PATH, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def build_knowledge_base_from_documents():
    """从文档目录构建知识库"""
    loader = DocumentLoader(DOCUMENTS_PATH)
    documents = loader.load_all_documents()

    if not documents:
        return False, "文档目录为空，请先上传文档"

    splitter = TextSplitter()
    split_docs = splitter.split_documents(documents)

    rag_chain = RAGChain()
    success = rag_chain.build_knowledge_base(split_docs)

    if success:
        st.session_state.rag_chain = rag_chain
        st.session_state.knowledge_base_built = True
        return True, f"知识库构建成功！共处理 {len(split_docs)} 个文本块"
    else:
        return False, "知识库构建失败"


def get_knowledge_base_status() -> dict:
    """获取知识库状态"""
    if st.session_state.rag_chain:
        return st.session_state.rag_chain.get_knowledge_base_status()
    elif os.path.exists(CHROMA_DB_PATH):
        return {"chunk_count": -1, "has_documents": True}
    return {"chunk_count": 0, "has_documents": False}


def display_chat_message(role: str, content: str, timestamp: str = None):
    """显示聊天消息"""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(f"**您:** {content}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**助手:** {content}")
            if timestamp:
                st.caption(f"时间: {timestamp}")


def main():
    st.set_page_config(
        page_title="RAG智能问答系统",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    st.title("📚 基于本地知识库的RAG智能问答系统")
    st.markdown("---")

    with st.sidebar:
        st.header("📁 文档管理")

        uploaded_files = st.file_uploader(
            "上传PDF或DOCX文档",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📤 添加到知识库", use_container_width=True):
                if uploaded_files:
                    with st.spinner("正在保存文件..."):
                        for uploaded_file in uploaded_files:
                            file_path = save_uploaded_file(uploaded_file)
                            st.success(f"已保存: {uploaded_file.name}")
                    st.info("文件已添加，请点击下方'🔄 构建知识库'按钮")
                else:
                    st.warning("请先上传文件")

        with col2:
            if st.button("🔄 构建知识库", use_container_width=True):
                with st.spinner("正在构建知识库..."):
                    success, message = build_knowledge_base_from_documents()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

        st.markdown("---")

        kb_status = get_knowledge_base_status()

        st.subheader("📊 知识库状态")
        if kb_status["has_documents"]:
            if kb_status["chunk_count"] >= 0:
                st.metric("文本块数量", kb_status["chunk_count"])
            else:
                st.metric("文本块数量", "已加载")
            st.success("✅ 知识库已就绪")
        else:
            st.metric("文本块数量", 0)
            st.info("📤 请上传文档并构建知识库")

        st.markdown("---")

        if st.button("🗑️ 清空对话历史", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()

    st.header("💬 问答交互")

    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            message.get("timestamp")
        )

    query = st.chat_input("请输入您的问题，按Enter键提交...")

    if query:
        timestamp = datetime.now().strftime("%H:%M:%S")

        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "timestamp": timestamp
        })

        display_chat_message("user", query, timestamp)

        if not st.session_state.knowledge_base_built:
            response = "⚠️ 知识库尚未构建，请先在左侧上传文档并点击'构建知识库'按钮"
        else:
            if st.session_state.rag_chain is None:
                st.session_state.rag_chain = RAGChain()
                st.session_state.rag_chain.initialize_chain()

            with st.spinner("正在思考..."):
                result = st.session_state.rag_chain.ask(
                    query,
                    st.session_state.chat_history
                )
                response = result["answer"]

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": query
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })

        response_timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": response_timestamp
        })

        display_chat_message("assistant", response, response_timestamp)

        with st.expander("📖 查看参考文档"):
            if st.session_state.knowledge_base_built and st.session_state.rag_chain:
                kb_status = get_knowledge_base_status()
                if kb_status["has_documents"]:
                    st.info("上方答案基于知识库中的文档生成")
                else:
                    st.info("暂无参考文档")
            else:
                st.info("知识库未构建")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "基于 Ollama + LangChain + Streamlit 构建 | "
        "使用 DeepSeek-R1 / Qwen2 模型"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()