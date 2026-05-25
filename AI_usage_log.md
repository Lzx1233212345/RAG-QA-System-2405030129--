# AI使用日志

本项目使用AI编程辅助工具（Trae）帮助生成代码骨架和调试错误。

## AI辅助生成的内容

### 1. 项目结构设计
- 使用LangChain + Chroma + Ollama的技术栈
- 模块化设计：文档加载、文本分割、向量存储、RAG链分离

### 2. 代码模块

| 文件 | AI生成内容 | 说明 |
|------|-----------|------|
| config.py | 全部 | 配置参数定义 |
| ollama_test.py | 全部 | Ollama连接测试 |
| document_loader.py | 全部 | 文档加载器 |
| text_splitter.py | 全部 | 文本分割器 |
| vector_store.py | 全部 | 向量存储管理 |
| rag_chain.py | 全部 | RAG问答链 |
| app.py | 全部 | Streamlit Web界面 |
| requirements.txt | 全部 | 依赖列表 |

### 3. 关键技术实现

#### 向量数据库选择
- **询问AI**: Chroma和FAISS相比各有什么优缺点？
- **AI回答**: Chroma更适合需要持久化和快速检索的场景，FAISS在内存中运行更快但不支持持久化
- **决策**: 选择Chroma，因为它便于调试和查看存储的数据

#### 文本分割策略
- **询问AI**: 如何选择合适的chunk_size和chunk_overlap？
- **AI回答**: chunk_size=1000适合大多数问题，overlap=200能保证上下文连贯性
- **决策**: 采用AI建议的参数

#### 提示词设计
- **询问AI**: 如何让RAG系统只在文档范围内回答，不知道时明确拒绝？
- **AI回答**: 在提示词中明确规则，并加入"文档中未找到相关答案"的处理逻辑
- **决策**: 采用AI建议的提示词模板

## 调试记录

### 问题1: Chroma向量数据库连接错误
- **现象**: 加载已存在的向量数据库时报错
- **询问AI**: Chroma持久化加载后无法进行相似性搜索
- **AI回答**: 需要在加载时指定embedding_function参数
- **解决**: 在VectorStoreManager.get_vectorstore()方法中添加embedding参数

### 问题2: RAG链返回答案不完整
- **现象**: 模型回答被截断
- **询问AI**: 如何让Ollama模型返回完整答案
- **AI回答**: 检查请求超时设置，可能需要增加timeout参数
- **解决**: 在ask()方法中捕获异常并优化错误处理

## AI使用心得

1. **有效利用**: AI最适合生成代码骨架和解决具体的技术问题
2. **需要验证**: AI生成的技术参数（如chunk_size）需要根据实际测试调整
3. **异常处理**: AI可能忽略边界情况，需要自行补充错误处理
4. **代码理解**: 使用AI生成代码后，需要理解每一部分的作用，便于后续维护

## 向AI提问的示例

### Q: 如何在Streamlit中实现多轮对话记忆？
A: 使用st.session_state存储对话历史，将历史消息传递给ConversationalRetrievalChain的chat_history参数。

### Q: Chroma向量数据库如何持久化存储？
A: 在创建Chroma时指定persist_directory参数，并调用.persist()方法保存。加载时使用相同的persist_directory即可。

### Q: 如何处理大文档的分块？
A: 使用RecursiveCharacterTextSplitter，设置适当的chunk_size和chunk_overlap，可基于段落或句子边界分割。