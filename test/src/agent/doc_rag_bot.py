import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from dotenv import load_dotenv

# 1. 加载配置
load_dotenv(encoding="utf-8")

# ==========================================
# 准备知识库 (读取本地 TXT 文件)
# ==========================================
# ==========================================
# 核心修改开始：动态获取绝对路径
# ==========================================

# 1. 获取当前脚本 (doc_rag_bot.py) 所在的目录
# 结果应该是 .../src/agent
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 拼接出 data_folder 的完整绝对路径
# 结果应该是 .../src/agent/data_folder
folder_path = os.path.join(current_dir, "data_folder")

# 调试：打印出来看看路径对不对
print(f"--- 正在尝试读取文件夹: {folder_path} ---")

# 3. 检查是否存在 (这一步能帮你确认路径到底对不对)
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"还是找不到文件夹！请确认该路径下存在文件夹: {folder_path}")

# ==========================================
# 核心修改结束
# ==========================================

print(f"--- 正在加载文件夹: {folder_path} ---")
# 1. 加载文档
# encoding="utf-8" 在 Windows 上必须加，否则读取中文会报错
print("--- 正在加载文件夹中的所有文档 ---")

# 定义一个加载器，告诉它去 'data_folder' 目录下找
# 【更高级的写法】如果你想精准控制不同后缀用不同加载器：
# 这种写法稍微复杂点，通常新手可以先手动一个个 load 然后 extend 到列表里
# 简单做法：
raw_docs = []

# 1. 读 txt
txt_loader = DirectoryLoader(
    folder_path,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    use_multithreading=True,
    show_progress=True
)
raw_docs.extend(txt_loader.load())

# 2. 读 pdf
pdf_loader = DirectoryLoader(
    folder_path,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    use_multithreading=True,
    show_progress=True
)
raw_docs.extend(pdf_loader.load())

# 3. 读 docx
docx_loader = DirectoryLoader(
    folder_path,
    glob="**/*.docx",
    loader_cls=Docx2txtLoader,
    use_multithreading=True,
    show_progress=True
)
raw_docs.extend(docx_loader.load())

print(f"--- 总共加载了 {len(raw_docs)} 个文档 ---")

# 2. 文本分割 (Splitting)
# 为什么分块？因为整本书太长，我们要把它切成小条，方便检索。
# chunk_size=200: 每块大约 200 个字符
# chunk_overlap=50: 每块之间重叠 50 个字 (防止关键信息刚好被切断)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "，"] # 优先按段落切，其次按句号切
)
docs = text_splitter.split_documents(raw_docs)

# 打印看看切分成了多少块
print(f"--- 文档已切分为 {len(docs)} 个片段 ---")

# 3. 初始化 Embedding (保持你刚才修改成功的配置)
# 如果你已经改成了 text2vec-base-chinese，这里不用动
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

# 4. 创建向量数据库 (这里也不用动)
print("--- 正在构建索引 ---")
vector_store = FAISS.from_documents(docs, embedding_model)
# k=5 确保能多查到几条相关信息
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# ==========================================
# 定义 LangGraph
# ==========================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


def customer_service_node(state: State):
    # 1. 获取用户最新的问题
    user_question = state["messages"][-1].content

    # 2. RAG 关键步骤：去向量库里搜索相关信息
    # 系统会自动计算 user_question 的向量，去匹配最接近的文档
    search_results = retriever.invoke(user_question)

    # 3. 把搜到的内容拼成字符串
    context_text = "\n\n".join([doc.page_content for doc in search_results])

    # 调试打印：看看它到底查到了什么 (在终端里看)
    print(f"--- 用户问: {user_question} ---")
    print(f"--- RAG查到: {context_text} ---")

    # 初始化模型 (请确保你配置了 OPENAI_API_KEY 环境变量)
    api_key = os.getenv("CHERRYSTUDIO_API_KEY")
    base_url = os.getenv("CHERRYSTUDIO_BASE_URL")
    llm = ChatOpenAI(
        model="deepseek/deepseek-v3.2(free)",  # 模型名字一定要对，看对应厂商的文档
        api_key=api_key,  # 将读到的 Key 传进去
        base_url=base_url  # 将读到的 URL 传进去)
    )

    # 5. 构建带有“上下文”的 Prompt
    # 告诉 AI：回答问题时，必须参考 Context 里的内容
    system_prompt = f"""
    你是一个手机店的智能客服。
    请根据下面的【参考资料】来回答用户的问题。
    如果参考资料里没有提到的内容，请诚实地说不知道，不要瞎编。

    【参考资料】：
    {context_text}
    """

    # 6. 发送给 AI
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)

    return {"messages": [response]}


# 构建图
builder = StateGraph(State)
builder.add_node("cs_agent", customer_service_node)
builder.add_edge(START, "cs_agent")
builder.add_edge("cs_agent", END)

graph = builder.compile()