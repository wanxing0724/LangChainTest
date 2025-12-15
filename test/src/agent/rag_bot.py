import os
from typing import Annotated
from langchain_huggingface import HuggingFaceEmbeddings
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# 1. 加载配置
load_dotenv(encoding="utf-8")

# ==========================================
# 准备知识库 (RAG 的核心数据)
# ==========================================
# 这里我们在内存里模拟一个简单的知识库，实际项目中通常是读取 PDF 或 TXT
knowledge_base_text = [
    "产品名称：未来手机 Pro (Future Phone Pro)",
    "价格：5999元人民币",
    "处理器：搭载最新的‘光速9000’芯片，运行速度提升 40%。",
    "电池：6000mAh 超大电池，支持 200W 快充，10分钟充满。",
    "屏幕：6.8英寸 4K 柔性屏，支持 144Hz 刷新率。",
    "保修政策：提供 2 年只换不修服务。",
    "竞争对手对比：比 iPhone 18 便宜 2000元，但性能更强。"
]

# 将文本转换为 Document 对象
docs = [Document(page_content=t) for t in knowledge_base_text]

# 初始化 Embedding 模型 (用于把文字变成向量)
# 注意：如果你用的是国内模型 key，这里可能需要换成 HuggingFaceEmbeddings(本地)
# 或者确保你的 base_url 提供商支持 /embeddings 接口
embedding_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

# 创建向量数据库 (Vector Store)
# 这步会把上面的 docs 变成向量存起来
print("--- 正在构建知识库索引 ---")
vector_store = FAISS.from_documents(docs, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # 每次只查最相关的 2 条


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