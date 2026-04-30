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
import json
from openai import AsyncOpenAI
import asyncio
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
#加载.env文件中的环境变量
load_dotenv()
#定义大模型调用函数，用于处理文本类模型生成功能
import os
from openai import OpenAI
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

async def LLM_replay_old(messages):
    """
    prompt_template:大模型调用的提示词模板
    message:大模型调用的用户输入
    """
    llm_client = AsyncOpenAI(api_key=os.getenv("QWEN_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    result = await llm_client.chat.completions.create(
        # 最新 qwen3-max
        model="qwen3-max", #推荐使用:qwen-plus-latest qwen-plus qwen-max-0125 qwen-plus-latest -0125
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
    png_path = './langGraph_basic_learning/output/graph_workflow.png'
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
    global work_app
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
    work_app = app

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
    # 默认会访问国外网站 mermaid.ink 生成图片，国内网络超时，所以报错。
    # 本地渲染，不联网
    graph_png = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
    img = mpimg.imread(BytesIO(graph_png), format='PNG')
    png_path = './langGraph_basic_learning/output/graph_workflow.png'
    plt.imsave(png_path, img)


def agent_flow():
    global agent_app
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
            "has_issues": true, 
            "issues": ["issue1", "issue2"],
            "severity": "low/medium/high",
            "confidence": 0.0-1.0
        }
        字段解释：
            has_issues： 是否含有敏感词
            issues：类型为列表，内容为字符串，必须是输入的内容，如果有敏感词，就填入，可以有多个，都是字符串，如果没有就为空列表
            severity：敏感等级，只能是这三个值low，medium和high
            confidence：置信度分数，为float类型，0到1之间
        比如：
        很敏感的结果：
        {
            "has_issues": true, 
            "issues": ["敏感词1", "敏感词2"],
            "severity": "high",
            "confidence": 1.0
        }
        返回的内容，必须json.load()能成功。
        """
        # langchain的格式
        # messages = [
        #     SystemMessage(content=system_prompt),
        #     HumanMessage(content=f"内容：{content}")
        # ]
        # 大模型的格式
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"内容：{content}"}
        ]
        

        response = LLM_replay(messages)
        try:
            result = json.loads(response)
            return {"analysis":result}
        except:
            # 解析失败，保守决策
            return {"analysis":""
            }

    def make_agent_decision(state: AgentModerationState) -> dict:
        """基于 LLM 分析做决策"""
        analysis = state["analysis"]
        content = state["content"]

        system_prompt = """你是内容审核决策助手， 基于之前的分析结果，做出审核决策：
        根据输入的内容，做出决策


        返回 JSON 格式：
        {
            "decision": "approved/rejected/needs_review",
            "reason": "简短说明",
            "confidence": 0.0-1.0
        }
        字段说明：
            decision：表示最后的决定，为字符串，有三个值依次为：
                - approved: 内容正常，通过
                - rejected: 明显违规，直接拒绝
                - needs_review: 需要人工审核
            reason：做出上述决策的原因
            confidence：做出上述决策的置信度
        比如，如果对一个内容很有信息是正常信息就如下回复：
        {
            "decision": "approved",
            "reason": "说明内容",
            "confidence": 1.0
        }
        返回的内容，必须json.load()能成功。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"原内容：{content}\n\n分析结果：{analysis}"}
        ]

        response = LLM_replay(messages)

        # 简化处理：直接解析响应（实际应用中应该用 JSON mode）
        try:
            result = json.loads(response)
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
    def should_auto_decide(state: AgentModerationState) -> Literal["decide", "human_review"]:
        """根据分析结果决定是否需要人工"""
        # 这里可以添加更复杂的逻辑
        # 例如：如果分析中提到高风险，直接转人工
        if dict == type(state.get("analysis")) and "has_issure" in state.get("analysis"):
            return "human_review"
        return "decide"

    def human_review_placeholder(state: AgentModerationState) -> dict:
        """人工审核占位符（实际应用中会中断等待人工）"""
        print("这里要人工审核", flush=True)
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
    # save 
    graph_png = agent_app.get_graph().draw_mermaid_png()
    img = mpimg.imread(BytesIO(graph_png), format='PNG')
    png_path = './langGraph_basic_learning/output/graph_llm_flow.png'
    plt.imsave(png_path, img)

    test_contents = [
        "这是一条正常的评论。",
        "我操你妈的，我日你祖宗十八代，干你娘",
        "AAAAAAAAAAAAAAAAAAA买买买！！！",
        "主席就是个垃圾"
    ]

    # === 测试 ===
    print("\n=== Agent 实现结果 ===\n")
    for content in test_contents:
        result = agent_app.invoke({"content": content})
        print(f"内容：{content}")
        print(f"分析：{result.get('analysis', 'N/A')}")
        print(f"决策：{result['decision']}")
        print(f"原因：{result['reason']}")
        print(f"置信度：{result.get('confidence', 'N/A')}\n")

# 混合方案
# 第一层：workflow 快速过滤明显的违规
# 第二层：Agent 处理边缘情况

def hybird_approch(content: str):
    # 1. 先调用workflow 快速检查
    workflow_result = work_app.invoke({"content": content})
    print("work_app result:{}".format(workflow_result))

    # 2. 如果workflow 决策明确（approved 或 rejected），直接返回
    if workflow_result['decision'] in ['approved', 'rejected']:
        return workflow_result

    # 3. 否则，使用Agent 做深度分析
    agent_result = agent_app.invoke({"content": content})
    print("agent_app result:{}".format(agent_result))
    return agent_result

if __name__=="__main__":
    # hello_world()    
    # branch()
    work_flow()
    print("==="*30)
    agent_flow()
    print("==="*30)
    print(hybird_approch("我就是闲聊，但是也聊工作，比较模糊，不好判断"))


