import streamlit as st
from src.agent.doc_rag_bot import graph  # 导入你做好的图

st.title("📱 未来手机 Pro - 智能客服")

# 1. 初始化聊天记录 (Streamlit 的 Session State)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 2. 显示历史消息
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 3. 处理用户输入
if user_input := st.chat_input("请输入您的问题..."):
    # 显示用户的话
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # --- 关键：调用你的 LangGraph ---
    # 构造输入
    inputs = {"messages": st.session_state["messages"]}

    # 调用模型 (stream_mode=False 简单点，直接拿结果)
    # 注意：这里 graph.invoke 会去执行你的 RAG 和 LLM
    result = graph.invoke(inputs)

    # 获取 AI 的最后一条回复
    ai_response = result["messages"][-1].content

    # --- 显示 AI 回复 ---
    with st.chat_message("assistant"):
        st.write(ai_response)
    st.session_state["messages"].append({"role": "assistant", "content": ai_response})