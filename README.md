# RAG智能问答系统

基于本地知识库的检索增强生成(RAG)智能问答系统，使用Ollama本地大模型、LangChain框架和Streamlit构建。

## 项目简介

本项目实现了一个能够"学习"本地文档并基于这些文档回答问题的智能问答系统。用户可以上传PDF或DOCX格式的文档，系统会自动解析、分块、向量化存储，当用户提问时，系统会从知识库中检索相关文档块，结合大模型生成准确答案。

## 环境要求

- **操作系统**: Windows 10/11
- **Python版本**: 3.10 或更高
- **Ollama**: 已安装并运行
- **模型**: DeepSeek-R1:7b 或 Qwen2:7b（需要提前下载）
- **内存**: 推荐16GB以上（用于运行大模型）

## 安装步骤

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 2. 安装并配置Ollama

访问 [Ollama官网](https://ollama.com/) 下载安装 Ollama for Windows。

安装完成后，在终端执行以下命令下载模型：

```bash
# 下载DeepSeek-R1 7B模型（推荐）
ollama pull deepseek-r1:7b

# 或者下载Qwen2 7B模型
ollama pull qwen2:7b

# 下载嵌入模型（用于文档向量化）
ollama pull nomic-embed-text
```

### 3. 验证Ollama服务

```bash
# 运行测试脚本
python ollama_test.py
```

确保看到 "Ollama服务连接成功" 的提示。

## 使用说明

### 启动Web应用

```bash
streamlit run app.py
```

应用启动后会自动在浏览器中打开界面（通常地址为 http://localhost:8501）。

### 上传文档

1. 在左侧边栏的"文档管理"区域
2. 点击"上传PDF或DOCX文档"按钮
3. 选择要上传的文档文件（支持多选）
4. 点击"📤 添加到知识库"按钮保存文件
5. 点击"🔄 构建知识库"按钮，系统会解析文档并构建向量索引

### 提问

1. 在底部的文本框中输入您的问题
2. 按Enter键或点击发送按钮提交问题
3. 系统会从知识库中检索相关文档并生成答案
4. 对话历史会显示在聊天区域

## 关键技术点说明

### RAG流程

1. **文档加载**: 使用PyPDFLoader和Docx2txtLoader加载PDF和DOCX文档
2. **文本分块**: 使用RecursiveCharacterTextSplitter进行智能分块（chunk_size=1000, chunk_overlap=200）
3. **向量化**: 使用Ollama内置的nomic-embed-text模型将文本块转换为向量
4. **存储**: 使用Chroma向量数据库存储和检索向量
5. **检索增强**: 根据用户问题检索最相关的文档块
6. **生成**: 将检索结果和问题发送给大模型生成答案

### 所用模型

- **主模型**: DeepSeek-R1:7b 或 Qwen2:7b（用于答案生成）
- **嵌入模型**: nomic-embed-text（用于文档向量化）

### 提示词设计

系统提示词要求模型：
- 只基于提供的参考文档回答
- 如果文档中没有相关信息，明确回复"文档中未找到相关答案"
- 保持回答准确、有条理

## 项目效果截图

### 图1：系统主界面
![系统主界面](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=RAG%20Question%20Answering%20System%20Web%20Interface%20with%20document%20upload%20panel%20and%20chat%20area%2C%20modern%20UI%20design%2C%20clean%20layout&image_size=landscape_16_9)

### 图2：文档上传与知识库构建
![文档上传](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Document%20upload%20interface%20for%20PDF%20and%20DOCX%20files%2C%20progress%20indicator%2C%20knowledge%20base%20building%2C%20modern%20web%20application&image_size=landscape_16_9)

### 图3：问答交互示例
![问答交互](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=AI%20chatbot%20conversation%20interface%20showing%20question%20and%20answer%20based%20on%20document%20knowledge%20base%2C%20chat%20history%2C%20modern%20UI&image_size=landscape_16_9)

### 问答示例

**问题**: 什么是自然语言处理？

**回答**: 自然语言处理（NLP）是人工智能的一个分支，主要研究如何使计算机能够理解、解释和生成人类语言。它涉及语音识别、文本分析、机器翻译、情感分析等多个领域，旨在实现人与计算机之间的自然语言交互。

**问题**: Transformer模型是如何工作的？

**回答**: Transformer模型基于自注意力机制，能够同时处理输入序列中的所有位置，捕捉不同位置之间的依赖关系。它由编码器和解码器组成，编码器负责理解输入序列，解码器负责生成输出序列，通过多头注意力机制实现并行计算，显著提升了模型的训练效率和性能。

## 项目结构

```
RAG-QA-System/
├── app.py              # Streamlit Web应用主文件
├── config.py            # 配置文件
├── document_loader.py  # 文档加载模块
├── text_splitter.py     # 文本分割模块
├── vector_store.py      # 向量存储模块
├── rag_chain.py         # RAG问答链模块
├── ollama_test.py       # Ollama连接测试脚本
├── requirements.txt     # Python依赖列表
├── README.md            # 项目说明文档
├── AI_usage_log.md      # AI使用日志
├── documents/           # 文档存储目录
└── chroma_db/           # 向量数据库目录
```

## 已知问题与改进方向

### 已知问题
1. 首次构建知识库时可能需要较长时间（取决于文档数量和大小）
2. 需要确保Ollama服务在后台运行

### 改进方向
1. 添加更多文档格式支持（如TXT、HTML等）
2. 实现批量上传功能
3. 添加导出问答记录功能
4. 支持夜间模式
5. 增加文档管理界面（查看已上传文档列表、删除文档等）

## 许可证

本项目仅供学习和研究使用.