"""
Ollama连接配置模块
AI辅助生成
"""
import os

# Ollama服务地址
OLLAMA_BASE_URL = "http://localhost:11434"

# 默认模型
DEFAULT_MODEL = "deepseek-r1:7b"

# 嵌入模型
EMBEDDING_MODEL = "nomic-embed-text"

# 向量数据库路径
CHROMA_DB_PATH = "./chroma_db"

# 文档存储路径
DOCUMENTS_PATH = "./documents"

# 文本分块参数
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 检索返回数量
TOP_K = 3