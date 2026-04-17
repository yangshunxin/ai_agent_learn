from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from io import BytesIO
import re
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
import asyncio
#加载.env文件中的环境变量
load_dotenv()
#定义大模型调用函数，用于处理文本类模型生成功能
async def LLM_replay(messages):
    """
    prompt_template:大模型调用的提示词模板
    message:大模型调用的用户输入
    """
    llm_client =AsyncOpenAI(api_key=os.getenv("QWEN_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    result =await llm_client.chat.completions.create(
        model="qwen-plus", #推荐使用:qwen-plus-latest qwen-plus qwen-max-0125 qwen-plus-latest -0125
        messages=[{"role":"user","content":messages}],
        max_tokens=8192,
        temperature=0)
    return result.choices[0].message.content

def hello_world():
    # 1. 定义state
    class HelloState(TypedDict):
        name: str
        greeting: str

    # 2. 定义节点
    def greet(state: HelloState) -> dict:
        name = state['name']
        return {'greeting': f"你好，{name}!"}

    def add_emoji(state: HelloState) -> dict:
        greeting = state['greeting']
        return {"greeting": greeting + " 👏"}

    # 3. 构建图
    graph = StateGraph(HelloState)

    # 第一个"greet"是名称，第二个是函数名称
    graph.add_node("greet", greet)
    graph.add_node("add_emoji", add_emoji)

    graph.add_edge(START, "greet")
    graph.add_edge("greet", "add_emoji")
    graph.add_edge("add_emoji", END)

    # 4. 编译
    app = graph.compile()

    # 5. 运行

    result = app.invoke({"name": "张三"})
    print(result)

    # 画图

    print(graph)

    graph_png = app.get_graph().draw_mermaid_png()
    img = mpimg.imread(BytesIO(graph_png), format='PNG')
    png_path = './langGraph_basic_learning/output/graph_hello_world.png'
    plt.imsave(png_path, img)

def branch():
    class WeatherState(TypedDict):
        temperature: int
        recommendation: str

    def check_temperature(state: WeatherState):
        # 这里可以调用真实的天气 API
        return {"temperature": 25}
    
    def route_by_temperature(state: WeatherState) -> Literal["cold", "warm", "hot"]:
        """根据温度路由，返回的是节点的名称"""
        temp = state["temperature"]
        if temp < 15:
            return "cold"
        elif temp < 25:
            return "warm"
        else:
            return "hot"
    
    def recommend_cold(state: WeatherState) -> dict:
        return {"recommendation": "穿厚外套！！"}
    
    def recommend_warm(state: WeatherState) -> dict:
        return {"recommendation": "穿长袖就好。"}
    
    def recommend_hot(state: WeatherState) -> dict:
        return {"recommendation": "穿短袖，记得防晒！"}
    
    # 构件图
    graph = StateGraph(WeatherState)

    graph.add_node("check", check_temperature)
    graph.add_node("cold", recommend_cold)
    graph.add_node("warm", recommend_warm)
    graph.add_node("hot", recommend_hot)
    
    graph.add_edge(START, "check")
    # 路由，
    graph.add_conditional_edges(
        "check",
        route_by_temperature,
        {
            # key: route_by_temperature函数 返回的字符串类型， value: 节点名称
            "cold": "cold",
            "warm": "warm",
            "hot": "hot"
        }
    )

    app = graph.compile()

    # 测试
    result = app.invoke({})
    print(result)

    # 保存
    graph_png = app.get_graph().draw_mermaid_png()
    img = mpimg.imread(BytesIO(graph_png), format='PNG')

    png_path = './langGraph_basic_learning/output/graph_branch.png'
    plt.imsave(png_path, img)


def work_flow():
    # state 定义
    class ContentModerationState(TypedDict):
        content: str
        has_profanity: bool
        has_spam_pattern: bool
        has_sensitive_topic: bool
        decision: str   # "approved" | "reject" | "need_review"
        reason: str
    
    # 节点实现
    def check_profanity(state: ContentModerationState) -> dict:
        """检查脏话（基于规则）"""
        content = state["content"].lower()
        # 实际应用中从数据库中加载
        profanity_list = ["脏话1", "脏话2", "敏感词"]

        has_profanity = any(word in content for word in profanity_list)
        return {"has_profanity": has_profanity}
    
    def check_spam(state: ContentModerationState) -> dict:
        """检查垃圾信息模式"""
        content = state["content"]

        # 规则1： 重复字符
        has_repeat = bool(re.search(r'(.)\1{5,}', content))

        # 规则2： 过多链接
        has_many_links = content.count("http") > 3

        # 规则3： 全大写
        has_all_caps = content.isupper() and len(content) > 20

        has_spam = has_repeat or has_many_links or has_all_caps

        return  {"has_spam_pattern": has_spam}
    
    def check_sensitive_topic(state: ContentModerationState) -> dict:
        """检查敏感话题"""
        content = state["content"].lower()
        sensitive_keywords = ["政治", "暴力", "色情"] # 简化实例，可以保存到数据库中

        has_sensitive = any(keyword in content  for keyword in sensitive_keywords)
        return {"has_sensitive_topic": has_sensitive}
    
    def make_decision(state: ContentModerationState) -> dict:
        """综合决策"""
        has_profanity = state.get("has_profanity", False)
        has_spam = state.get("has_spam_pattern", False)
        has_sensitive = state.get("has_sensitive_topic", False)
        if has_profanity:
            return {
                "decision": "reject",
                "reason": "包含不当语言"
            }
        elif has_sensitive:
            return {
                "decision": "needs_review",
                "reason":"包含敏感话题，需人工审核"
            }
        elif has_spam:
            return {
                "decision": "rejected",
                "reason":"疑似垃圾信息"
            }
        else:
            return {
                "decision":"approved",
                "reason":"内容正常"
            }
        
    # 构建flow
    workflow_graph = StateGraph(ContentModerationState)

    #添加节点
    workflow_graph.add_node("check_profanity", check_profanity)
    workflow_graph.add_node("check_spam", check_spam)
    workflow_graph.add_node("check_sensitive", check_sensitive_topic)
    workflow_graph.add_node("decide", make_decision)

    # 固定的执行顺序
    workflow_graph.add_edge(START, "check_profanity")
    workflow_graph.add_edge("check_profanity", "check_spam")
    workflow_graph.add_edge("check_spam", "check_sensitive")
    workflow_graph.add_edge("check_sensitive", "decide")
    workflow_graph.add_edge("decide", END)

    # 编译
    app = workflow_graph.compile()

    # test
    result = app.invoke({"content": "随便聊聊"})
    print(result)
    result = app.invoke({"content": "这里有敏感词"})
    print(result)
    result = app.invoke({"content": "AAAAAAAAAAAAAAAAAAA"})
    print(result)
    result = app.invoke({"content": "http:1345454, http:/fafafafa, http:///fafafa, http"})
    print(result)
    result = app.invoke({"content": "色情内容信息，政治暴力"})
    print(result)
    # save
    graph_png = app.get_graph().draw_mermaid_png()
    img = mpimg.imread(BytesIO(graph_png), format='PNG')
    png_path = './langGraph_basic_learning/output/graph_workflow.png'
    plt.imsave(png_path, img)


def agent_flow():
    # === State 定义 ===
    class AgentModerationState(TypedDict):
        content: str
        analysis: str
        decision: str  # "approved" | "rejected" | "needs_review"
        reason: str
        confidence: float
    # === LLM 初始化 ===
    # 见上

    # === 节点实现 ===

    def analyze_content(state: AgentModerationState) -> dict:
        """使用 LLM 分析内容"""
        content = state["content"]

        system_prompt = """你是一个内容审核助手。分析给定的内容，判断是否包含：
        1. 不当语言（脏话、侮辱）
        2. 垃圾信息（广告、刷屏）
        3. 敏感话题（政治、暴力、色情）

        请以 JSON 格式返回：
        {
            "has_issues": true/false,
            "issues": ["issue1", "issue2"],
            "severity": "low/medium/high",
            "confidence": 0.0-1.0
        }
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"内容：{content}")
        ]
        

        response = await LLM_replay(messages)
        return {"analysis": response.content}

    def make_agent_decision(state: AgentModerationState) -> dict:
        """基于 LLM 分析做决策"""
        analysis = state["analysis"]
        content = state["content"]

        system_prompt = """基于之前的分析结果，做出审核决策：
        - approved: 内容正常，通过
        - rejected: 明显违规，直接拒绝
        - needs_review: 需要人工审核

        返回 JSON 格式：
        {
            "decision": "approved/rejected/needs_review",
            "reason": "简短说明",
            "confidence": 0.0-1.0
        }
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"原内容：{content}\n\n分析结果：{analysis}")
        ]

        response = model.invoke(messages)

        # 简化处理：直接解析响应（实际应用中应该用 JSON mode）
        import json
        try:
            result = json.loads(response.content)
            return {
                "decision": result["decision"],
                "reason": result["reason"],
                "confidence": result["confidence"]
            }
        except:
            # 解析失败，保守决策
            return {
                "decision": "needs_review",
                "reason": "分析结果解析失败",
                "confidence": 0.0
            }
    # === 路由函数 ===
    def should_auto_decide(state: AgentModerationState) -> Literal["decide", "review"]:
        """根据分析结果决定是否需要人工"""
        # 这里可以添加更复杂的逻辑
        # 例如：如果分析中提到高风险，直接转人工
        if "high" in state.get("analysis", "").lower():
            return "review"
        return "decide"

    def human_review_placeholder(state: AgentModerationState) -> dict:
        """人工审核占位符（实际应用中会中断等待人工）"""
        return {
            "decision": "needs_review",
            "reason": "已转人工审核",
            "confidence": 1.0
        }

    # === 构建 Agent Graph ===
    agent_graph = StateGraph(AgentModerationState)

    agent_graph.add_node("analyze", analyze_content)
    agent_graph.add_node("decide", make_agent_decision)
    agent_graph.add_node("human_review", human_review_placeholder)

    # === 构建 Agent Graph ===
    agent_graph = StateGraph(AgentModerationState)

    agent_graph.add_node("analyze", analyze_content)
    agent_graph.add_node("decide", make_agent_decision)
    agent_graph.add_node("human_review", human_review_placeholder)

    agent_graph.add_edge(START, "analyze")
    agent_graph.add_conditional_edges(
        "analyze",
        should_auto_decide,
        {
            "decide": "decide",
            "review": "human_review"
        }
    )
    agent_graph.add_edge("decide", END)
    agent_graph.add_edge("human_review", END)

    agent_app = agent_graph.compile()

    test_contents = [
        "这是一条正常的评论。",
        "这包含脏话1的内容",
        "AAAAAAAAAAAAAAAAAAA买买买！！！",
        "讨论政治话题的内容"
    ]

    # === 测试 ===
    print("\n=== Agent 实现结果 ===\n")
    for content in test_contents:
        result = agent_app.invoke({"content": content})
        print(f"内容：{content}")
        print(f"分析：{result.get('analysis', 'N/A')[:100]}...")
        print(f"决策：{result['decision']}")
        print(f"原因：{result['reason']}")
        print(f"置信度：{result.get('confidence', 'N/A')}\n")

if __name__=="__main__":
    # hello_world()    
    # branch()
    work_flow()


