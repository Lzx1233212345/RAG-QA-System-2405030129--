"""
RAG智能问答系统 - Streamlit Web应用
基于本地知识库的智能问答系统
"""
import streamlit as st
import os
import shutil
from datetime import datetime
from typing import List
import requests

# 导入本地模块
try:
    from config import DOCUMENTS_PATH, CHROMA_DB_PATH
    from document_loader import DocumentLoader
    from text_splitter import TextSplitter
    from rag_chain import RAGChain
except ImportError as e:
    st.error(f"导入错误: {str(e)}")
    st.stop()

# 确保目录存在
os.makedirs(DOCUMENTS_PATH, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# 初始化会话状态
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

initialize_session_state()

# 页面配置
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="🤖",
    layout="wide"
)

# 页面标题
st.title("🤖 RAG智能问答系统")
st.markdown("基于本地知识库的智能问答系统")

# 侧边栏 - 文档管理
with st.sidebar:
    st.header("📁 文档管理")
    
    # 文件上传
    uploaded_files = st.file_uploader(
        "上传PDF或DOCX文档",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DOCUMENTS_PATH, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ 已保存: {uploaded_file.name}")
    
    # 显示已上传文档
    doc_count = len([f for f in os.listdir(DOCUMENTS_PATH) if f.endswith((".pdf", ".docx"))])
    st.info(f"📄 当前文档数: {doc_count}")
    
    # 构建知识库按钮
    if st.button("🔄 构建知识库", type="primary"):
        if doc_count == 0:
            st.warning("⚠️ 请先上传文档！")
        else:
            with st.spinner("正在构建知识库..."):
                try:
                    # 初始化组件
                    loader = DocumentLoader(DOCUMENTS_PATH)
                    splitter = TextSplitter()
                    rag_chain = RAGChain()
                    
                    # 加载文档
                    st.write("📖 正在加载文档...")
                    documents = loader.load_all_documents()
                    
                    if not documents:
                        st.error("❌ 未能加载任何文档！")
                    else:
                        # 分块
                        st.write("✂️ 正在分割文档...")
                        split_docs = splitter.split_documents(documents)
                        
                        # 构建向量存储
                        st.write("🧠 正在构建向量数据库...")
                        rag_chain.build_knowledge_base(split_docs)
                        
                        # 保存到会话状态
                        st.session_state.rag_chain = rag_chain
                        st.session_state.knowledge_base_built = True
                        
                        st.success(f"✅ 知识库构建完成！共处理了 {len(split_docs)} 个文本块")
                except Exception as e:
                    st.error(f"构建知识库时出错: {str(e)}")
    
    # 知识库状态
    st.divider()
    st.subheader("📊 知识库状态")
    
    if st.session_state.knowledge_base_built:
        st.success("✅ 知识库已就绪")
        if st.session_state.rag_chain:
            status = st.session_state.rag_chain.get_knowledge_base_status()
            st.info(f"📄 文本块数: {status['chunk_count']}")
    else:
        st.warning("⚠️ 请先构建知识库")

# 主聊天界面
st.divider()

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入
if prompt := st.chat_input("请输入你的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 生成回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if not st.session_state.knowledge_base_built:
            message_placeholder.markdown("⚠️ 知识库尚未构建，请先上传文档并构建知识库！")
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ 知识库尚未构建，请先上传文档并构建知识库！"})
        else:
            with st.spinner("正在思考..."):
                try:
                    result = st.session_state.rag_chain.ask(prompt, st.session_state.chat_history)
                    message_placeholder.markdown(result["answer"])
                    
                    # 更新对话历史
                    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
                    
                    # 显示参考文档
                    if result["source_documents"]:
                        with st.expander("📚 参考文档"):
                            for i, doc in enumerate(result["source_documents"]):
                                st.markdown(f"**文档 {i+1}:**")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                except Exception as e:
                    error_msg = f"❌ 处理问题时出错: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# 页脚
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>RAG智能问答系统 | 基于Ollama + LangChain + Streamlit</p>
</div>
""", unsafe_allow_html=True)