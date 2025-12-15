from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
# 如果你用的是其他模型（比如 Anthropic），请导入对应的包
import os

# 【重点1】加载环境变量
# 虽然 langgraph dev 通常会自动加载 .env，但为了保险，
# 或者是以后你直接运行脚本，建议加上这就行：
from dotenv import load_dotenv
load_dotenv(encoding="utf-8")
# 1. 定义状态 (State)


# State 是 LangGraph 的核心，它就像机器人的“短期记忆”
# 这里我们定义一个简单的状态，只包含一个 "messages" 列表
class State(TypedDict):
    # add_messages 的作用是：新的消息会追加到列表中，而不是覆盖，存储所有的聊天记录
    messages: Annotated[list, add_messages]

# 2. 定义节点 (Nodes)
# 节点就是图中的圆圈，本质上就是普通的 Python 函数
def chatbot(state: State):
    # 初始化模型 (请确保你配置了 OPENAI_API_KEY 环境变量)
    api_key = os.getenv("CHERRYSTUDIO_API_KEY")
    base_url = os.getenv("CHERRYSTUDIO_BASE_URL")
    llm = ChatOpenAI(
        model="deepseek/deepseek-v3.2(free)", # 模型名字一定要对，看对应厂商的文档
        api_key=api_key,       # 将读到的 Key 传进去
        base_url=base_url      # 将读到的 URL 传进去)
    )
    # C. 【关键步骤】定义“人设” (System Prompt)
    # 这段话不会发给用户，但会告诉 AI 怎么做
    system_prompt = SystemMessage(content="""
        你是一家名为“未来科技”的手机店的智能客服。

        你的职责：
        1. 热情地回答用户关于手机参数、价格和售后的问题。
        2. 如果用户问竞争对手（如苹果、三星）的问题，你要委婉地把话题引回到我们自家的“未来手机 Pro”上。
        3. 如果用户问数学题、编程题或其他无关话题，请礼貌拒绝，说你只负责手机咨询。
        4. 语气要活泼，多用表情符号 😊。
        """)

    # D. 拼接消息
    # 我们把“系统人设”放在最前面，后面跟着“历史聊天记录”
    # 注意：我们这里构造一个新的列表传给 LLM，而不是把 SystemMessage 存进 state 里的数据库
    # 这样可以避免每次对话都重复存一遍人设，节省 Token。
    messages_to_send = [system_prompt] + state["messages"]

    # E. 调用模型
    response = llm.invoke(messages_to_send)

    # F. 返回结果 (LangGraph 会自动把它追加到历史记录里)
    return {"messages": [response]}

# 3. 构建图 (Graph)
graph_builder = StateGraph(State)

# 添加节点，给它起个名字叫 "chatbot"
graph_builder.add_node("chatbot", chatbot)

# 添加边 (Edges)，定义流程
# 从 START (开始) -> 走到 "chatbot" 节点
graph_builder.add_edge(START, "chatbot")
# 从 "chatbot" 节点 -> 走到 END (结束)
graph_builder.add_edge("chatbot", END)

# 4. 编译图
# 这就是 LangGraph Studio 最终加载的对象
graph = graph_builder.compile()