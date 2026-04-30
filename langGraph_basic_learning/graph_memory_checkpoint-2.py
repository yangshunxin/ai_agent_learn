from typing import TypedDict, Literal,Annotated
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from io import BytesIO
import re
from dotenv import load_dotenv
import os
import json
from openai import AsyncOpenAI
import asyncio
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages, Messages
from langgraph.errors import GraphInterrupt
from operator import add
import time
#加载.env文件中的环境变量
load_dotenv()
#定义大模型调用函数，用于处理文本类模型生成功能
import os
from openai import OpenAI
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver 
from dashscope import Generation
import dashscope
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

work_app = None
agent_app = None

def LLM_replay(messages):
    response =  Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-turbo",
        messages=messages,
        result_format="message",
        # 开启深度思考
        enable_thinking=False,
    )

    if response.status_code == 200:
        # # 打印思考过程
        # print("=" * 20 + "思考过程" + "=" * 20)
        # print(response.output.choices[0].message.reasoning_content)
        
        # # 打印回复
        # print("=" * 20 + "完整回复" + "=" * 20)
        # print(response.output.choices[0].message.content)
        # return response.output.choices[0].message.content[0]['text']
        return response.output.choices[0].message.content
    else:
        # print(f"HTTP返回码：{response.status_code}")
        # print(f"错误码：{response.code}")
        # print(f"错误信息：{response.message}")
        return "call llm error"

# 公共部分
class SimpleState(TypedDict):
    count: int
    messages: Annotated[Messages, add_messages]

def increment(state: SimpleState) -> dict:
    return {"count": state.get("count", 0) + 1}

# 创建图
graph = StateGraph(SimpleState)
graph.add_node("increment", increment)
graph.add_edge(START, "increment")
graph.add_edge("increment", END)

app = graph.compile()

def simple_memory_saver():


    # 使用 MemorySaver
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    # 运行时指定 thread_id
    config = {"configurable": {"thread_id": "thread-1"}}

    # 第一次运行
    result1 = app.invoke({"count": 0}, config)
    print(result1)  # {"count": 1}

    # 第二次运行（同一个 thread，状态会累加）
    result2 = app.invoke({}, config)
    print(result2)  # {"count": 2}

    # 新的 thread（独立的状态）
    config2 = {"configurable": {"thread_id": "thread-2"}}
    result3 = app.invoke({"count": 0}, config2)
    print(result3)  # {"count": 1}


# 持久化可以后续测试，比如：数据库和redis
def thread_test():
        # 使用 MemorySaver
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    # 用户 A 的会话
    config_user_a = {"configurable": {"thread_id": "user-a-session-1"}}
    print(app.invoke({"messages": [{"role": "user", "content": "您好"}]}, config_user_a))

    # 用户 B 的会话（完全独立）
    config_user_b = {"configurable": {"thread_id": "user-b-session-1"}}
    print(app.invoke({"messages": [{"role": "user", "content": "Hi!"}]}, config_user_b))

    # 用户 A 继续会话（状态保持）
    print(app.invoke({"messages": [{"role": "user", "content": "我之前说了什么"}]}, config_user_a))
    # LLM 能看到 "你好"
    print("**"*20)
    # 获取某个 thread 的所有 checkpoints
    history = app.get_state_history(config_user_a)
    for checkpoint in history:
        print(f"Checkpoint ID: {checkpoint.config['configurable']['checkpoint_id']}")
        print(f"State: {checkpoint.values}")
        print(f"Next: {checkpoint.next}")
        print("---")

def test_continue_run():
    class PipelineState(TypedDict):
        total_items: int
        processed_items: Annotated[int, add]
        failed_items: Annotated[list, lambda old, new: old + new]
        results: Annotated[list, lambda old, new: old + new]

    # 模拟步骤
    def fetch_data(state: PipelineState) -> dict:
        """步骤 1：获取数据"""
        print("📥 获取数据...")
        time.sleep(1)
        return {"total_items": 100}

    def process_batch_1(state: PipelineState) -> dict:
        """步骤 2：处理批次 1"""
        print("⚙️  处理批次 1 (items 1-33)...")
        time.sleep(2)
        return {
            "processed_items": 33,
            "results": [f"result-{i}" for i in range(1, 34)]
        }

    def process_batch_2(state: PipelineState) -> dict:
        """步骤 3：处理批次 2"""
        print("⚙️  处理批次 2 (items 34-66)...")
        time.sleep(2)

        # 模拟失败（第一次运行时）
        if state["processed_items"] == 33:
            raise Exception("💥 批次 2 处理失败！（模拟错误）")

        return {
            "processed_items": 33,
            "results": [f"result-{i}" for i in range(34, 67)]
        }

    def process_batch_3(state: PipelineState) -> dict:
        """步骤 4：处理批次 3"""
        print("⚙️  处理批次 3 (items 67-100)...")
        time.sleep(2)
        return {
            "processed_items": 34,
            "results": [f"result-{i}" for i in range(67, 101)]
        }

    def finalize(state: PipelineState) -> dict:
        """步骤 5：完成"""
        print(f"✅ 完成！总共处理 {state['processed_items']} 项")
        return {}

    # 构建图
    graph = StateGraph(PipelineState)
    graph.add_node("fetch", fetch_data)
    graph.add_node("batch_1", process_batch_1)
    graph.add_node("batch_2", process_batch_2)
    graph.add_node("batch_3", process_batch_3)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", "batch_1")
    graph.add_edge("batch_1", "batch_2")
    graph.add_edge("batch_2", "batch_3")
    graph.add_edge("batch_3", "finalize")
    graph.add_edge("finalize", END)

    # 使用 SqliteSaver
    # checkpointer = SqliteSaver.from_conn_string("pipeline.db")
    checkpointer = SqliteSaver(sqlite3.connect("pipeline.db", check_same_thread=False))

    app = graph.compile(checkpointer=checkpointer)

    # 配置
    config = {"configurable": {"thread_id": "pipeline-1"}}

    # === 第一次运行（会失败） ===
    print("\n" + "="*50)
    print("第一次运行（会在批次 2 失败）")
    print("="*50 + "\n")

    try:
        result = app.invoke({"processed_items": 0, "results": []}, config)
    except Exception as e:
        print(f"\n❌ 运行失败：{e}\n")
        print("但是批次 1 的进度已保存！\n")

    # 查看 checkpoint
    state = app.get_state(config)
    print(f"已保存的状态：")
    print(f"  - 已处理：{state.values['processed_items']} 项")
    print(f"  - 下一个节点：{state.next}\n")

    # === 修复后重新运行 ===
    print("="*50)
    print("修复后重新运行（从批次 2 继续）")
    print("="*50 + "\n")

    # 修改 process_batch_2 不再失败（这里我们直接修改状态模拟修复）
    # 实际应用中，你会修复代码然后重新部署

    # 再次运行 - 会从批次 2 继续！
    result = app.invoke({}, config)

    print(f"\n✅ 最终结果：")
    print(f"  - 总项目：{result['total_items']}")
    print(f"  - 已处理：{result['processed_items']}")
    print(f"  - 结果数量：{len(result['results'])}")

if __name__=="__main__":
    # simple_memory_saver()
    # print("=="*20)
    # thread_test()
    print("=="*20)
    test_continue_run()




