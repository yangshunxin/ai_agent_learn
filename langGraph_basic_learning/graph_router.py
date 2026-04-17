from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

# ===================== 1. 定义工具 =====================
@tool
def check_at_home(user_name: str) -> str:
    """检查用户是否在家"""
    if user_name == "张三":
        return "在家"
    else:
        return "不在家"

tools = [check_at_home]
tool_node = ToolNode(tools)

# ===================== 2. 定义LLM =====================
llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)

# ===================== 3. 图节点 =====================
def call_llm(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 分支A：在家时执行
def at_home_node(state: MessagesState):
    return {"messages": [AIMessage(content="✅ 用户在家，可以上门服务")]}

# 分支B：不在家时执行
def not_at_home_node(state: MessagesState):
    return {"messages": [AIMessage(content="❌ 用户不在家，改天再联系")]}

# ===================== 4. 【核心】条件路由函数 =====================
def route_based_on_tool_result(state: MessagesState):
    """
    根据工具返回结果 选择分支
    返回值 = 下一个要走的节点名称
    """
    messages = state["messages"]
    last_message = messages[-1]  # 工具返回的消息一定是最后一条

    # 获取工具返回内容
    tool_result = last_message.content.strip()

    # 根据结果路由
    if tool_result == "在家":
        return "at_home_node"  # 去分支A
    elif tool_result == "不在家":
        return "not_at_home_node"  # 去分支B
    else:
        return END

# ===================== 5. 构建流程图 =====================
workflow = StateGraph(MessagesState)

# 添加节点
workflow.add_node("llm", call_llm)
workflow.add_node("tools", tool_node)
workflow.add_node("at_home_node", at_home_node)
workflow.add_node("not_at_home_node", not_at_home_node)

# 设置入口
workflow.set_entry_point("llm")

# 第一步路由：是否调用工具
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges(
    "llm",
    should_continue,
    ["tools", END]
)

# 【核心】第二步路由：工具执行完 → 根据结果选分支
workflow.add_conditional_edges(
    "tools",  # 从工具节点出发
    route_based_on_tool_result,  # 路由函数
    {
        "at_home_node": "at_home_node",
        "not_at_home_node": "not_at_home_node"
    }
)

# 分支执行完直接结束
workflow.add_edge("at_home_node", END)
workflow.add_edge("not_at_home_node", END)

# 编译图
graph = workflow.compile()

# ===================== 测试 =====================
if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="查一下张三在不在家")]}
    result = graph.invoke(inputs)
    print(result["messages"][-1].content)