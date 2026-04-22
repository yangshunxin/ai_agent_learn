"""
一个 MCP client实现多个 MCP server连接+调用实战
pip install openai
pip install mcp
pip install matplotlib
pip install dotenv
"""
import json
from mcp.client.sse import sse_client  #sse_client
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncOpenAI
from dotenv import load_dotenv
from contextlib import AsyncExitStack
import asyncio
from mcp import ClientSession
import sys
import os
import uvicorn
import time
from pathlib import Path
from contextlib import asynccontextmanager

# ====================== Web 服务（FastAPI + 标准 OpenAI 接口）======================
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# 加载环境变量
load_dotenv()

#加载MCP服务器的地址
server_url = os.getenv("server_url")

MODULE_NAME=os.getenv("module_name")
print("module_name:{}".format(MODULE_NAME))

# ====================== 模型配置（可自由切换）======================
MODEL_BACKEND = os.getenv("model_name")

# OpenAI 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Ollama 配置（本地模型）
OLLAMA_API_KEY = "ollama"  # 固定无需修改
OLLAMA_BASE_URL = "http://{}:11434/v1".format(server_url)
OLLAMA_MODEL = "qwen3:latest"  # 你本地的模型名 qwen llama3 gemma 都行

#通用MCP连接管理类
class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions = {}
        self.tools_map = {}

        # 自动根据 MODEL_BACKEND 初始化客户端
        if MODEL_BACKEND == "deepseek":
            self.client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
            self.model = DEEPSEEK_MODEL
        elif MODEL_BACKEND == "ollama":
            self.client = AsyncOpenAI(api_key=OLLAMA_API_KEY, base_url=OLLAMA_BASE_URL)
            self.model = OLLAMA_MODEL
        else:
            raise ValueError("MODEL_BACKEND 必须是 openai 或 ollama")

    async def connect_to_server(self, server_id: str, endpoint_url: str, protocal_type: str):
        if server_id in self.sessions:
            raise ValueError(f"服务端 {server_id} 已连接")

        if protocal_type == "sse":
            transport = await self.exit_stack.enter_async_context(
                sse_client(endpoint_url, timeout=10000, sse_read_timeout=10000)
            )
            read_stream, write_stream = transport
        else:
            transport = await self.exit_stack.enter_async_context(
                streamablehttp_client(endpoint_url, timeout=10000, sse_read_timeout=10000)
            )
            read_stream, write_stream, _ = transport

        session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        self.sessions[server_id] = {
            "session": session,
            "read_stream": read_stream,
            "write_stream": write_stream,
            "endpoint_url": endpoint_url,
            "protocal_type": protocal_type
        }

        # 同步工具列表
        resp = await session.list_tools()
        for tool in resp.tools:
            self.tools_map[tool.name] = (server_id, endpoint_url)

    async def list_all_tools(self):
        tools = []
        for server_id in self.sessions:
            sess = self.sessions[server_id]["session"]
            resp = await sess.list_tools()
            for t in resp.tools:
                safe_name = t.name.replace('-', '_')
                tools.append({
                    "type": "function",
                    "function": {
                        "name": safe_name,
                        "description": t.description,
                        "parameters": t.inputSchema
                    }
                })
        return tools
    
    async def process_query(self, messages: list) -> dict:
        """
        标准 OpenAI 格式输出，可直接用于 WebUI
        返回: { "content": str, "tool_calls": list, "full_history": list }
        """
        print(f"\n======================================")
        print(f"🔵 开始处理用户请求")
        print(f"📩 输入消息: {json.dumps(messages, ensure_ascii=False, indent=2)}")

        tools = await self.list_all_tools()
        print(f"🔧 加载可用工具数量: {len(tools)}")
        if tools:
            print(f"📋 工具列表: {json.dumps(tools, ensure_ascii=False, indent=2)}")

        # ===================== 自动注入系统 Prompt（图片自动显示）=====================
        system_prompt = f"""
你是结果优化助手。
规则：
1. 如果你调用画图工具后返回了以http开头，并且以.png / .jpg / .svg 结尾的图片链接，
   必须使用 Markdown 图片格式输出，让 Open WebUI 直接显示图片：
   ![图表](图片链接)

2. 如果是文字结果，正常回答即可，图片链接必须完整。
"""
        # 在消息历史最前面插入系统提示
        history = [{"role": "system", "content": system_prompt}] + messages.copy()
        # =============================================================================
        
        round_count = 0

        while True:
            round_count += 1
            print(f"\n---------------- 第 {round_count} 轮调用大模型 ----------------")

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=history,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                max_tokens=4096,
                temperature=0,
                stream=False
            )

            choice = response.choices[0]
            msg = choice.message

            print(f"🤖 模型原始返回: {str(msg)}")
            print(f"✅ 结束原因: {choice.finish_reason}")

            if choice.finish_reason == "tool_calls":
                print(f"⚙️ 模型需要调用工具")

                history.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": msg.tool_calls
                })

                for call in msg.tool_calls:
                    tool_name = call.function.name
                    args = json.loads(call.function.arguments)

                    print(f"\n⚙️ 准备调用工具: {tool_name}")
                    print(f"📥 工具参数: {json.dumps(args, ensure_ascii=False, indent=2)}")

                    # 还原原名（下划线 → 横杠）
                    original_name = tool_name
                    if original_name not in self.tools_map:
                        original_name = tool_name.replace('_', '-')
                        print(f"🔄 工具名映射为: {original_name}")

                    server_id, _ = self.tools_map[original_name]
                    print(f"🖥️ 来自MCP服务: {server_id}")

                    # ===================== 计时开始 =====================
                    start_time = time.time()

                    sess = self.sessions[server_id]["session"]
                    result = await sess.call_tool(original_name, args)
                    content = result.content[0].text

                    # ===================== 计时结束 =====================
                    duration = time.time() - start_time
                    print(f"⏱️ 工具【{original_name}】执行耗时: {duration:.3f} 秒")

                    print(f"📤 工具执行结果: {content}")

                    history.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": content
                    })

            else:
                print(f"\n🏁 最终AI回复: {msg.content}")
                print(f"======================================\n")

                return {
                    "content": msg.content,
                    "tool_calls": None,
                    "full_history": history
                }
            
    async def clean(self):
        """清理所有资源"""
        self.exit_stack.aclose()
        self.sessions.clear()
        self.tools_map.clear()


mcp_client = MCPClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行（等价于 startup）
    await mcp_client.connect_to_server(
        "search_db_mcp",
        f"http://{server_url}:9004/mcp",
        "streamable-http"
    )
    await mcp_client.connect_to_server(
        "python_chart_mcp",
        f"http://{server_url}:9002/mcp",
        "streamable-http"
    )
    await mcp_client.connect_to_server(
        "machine_learning_mcp",
        f"http://{server_url}:9003/mcp",
        "streamable-http"
    )
    print("✅ 所有 MCP 服务器连接完成")
    tools = await mcp_client.list_all_tools()
    print(f"✅ 加载工具数量：{len(tools)}")

    yield  # 服务运行中

    # 关闭时执行（等价于 shutdown）
    await mcp_client.clean()
    print("🛑 服务已关闭，资源已释放")

# 把 lifespan 传入 FastAPI
app = FastAPI(
    title="MCP + OpenAI/Ollama = AI for BI",
    version="1.0",
    lifespan=lifespan  # 关键在这里
)
# 跨域（WebUI 必备）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 本地系统固定路径（你真实存图片的地方）
image_dir = os.getenv("IMAGE_DIR")
IMAGE_STORAGE_DIR = Path(image_dir)
IMAGE_STORAGE_DIR.mkdir(exist_ok=True)

server_port = os.getenv("server_port")

# 2. URL 访问路径（固定写 /images 就行！）
IMAGE_URL_PREFIX = "/images"

# 把 images 文件夹变成外网可访问
# app.mount("/URL路径", StaticFiles(directory="本地硬盘路径"))
app.mount(IMAGE_URL_PREFIX, StaticFiles(directory=IMAGE_STORAGE_DIR), name="images")
image_prefix = "http://{}:{}/images/".format(server_url, server_port)

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: Optional[str] = None

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def openai_chat_completions(
    req: ChatRequest,
    request: Request  # 用来获取IP和请求路径
):
    # ===================== 日志：请求信息 =====================
    client_ip = request.client.host
    path = request.url.path
    print(f"\n==================================================")
    print(f"📥 收到聊天请求 | IP: {client_ip} | 路径: {path}")
    print(f"📩 用户消息: {json.dumps(req.messages, ensure_ascii=False)}")
    # =========================================================

    # 执行核心逻辑
    result = await mcp_client.process_query(req.messages)

    # 构建返回
    response = {
        "id": "mcp-chat-001",
        "object": "chat.completion",
        "model": mcp_client.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["content"]
                },
                "finish_reason": "stop"
            }
        ]
    }

    # ===================== 日志：返回结果 =====================
    print(f"📤 AI 响应: {json.dumps(response, ensure_ascii=False, indent=2)}")
    print(f"==================================================\n")
    # ==========================================================

    return response

# 同时支持 /v1/models 和 /models，彻底解决 OpenWebUI 404
@app.get("/v1/models")
@app.get("/models")
async def list_models(request: Request):
    
    # ===================== 自动日志打印 =====================
    client_ip = request.client.host
    path = request.url.path
    print(f"📥 收到模型列表请求 | IP: {client_ip} | 路径: {path}")
    # =======================================================

    response = {
        "data": [
            {
                "id": MODULE_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "mcp-service",
                "permission": []
            }
        ],
        "object": "list"
    }

    # ===================== 打印返回结果 =====================
    print(f"📤 返回模型列表: {json.dumps(response, ensure_ascii=False, indent=2)}")
    # ========================================================

    return response
    

@app.get("/health")
async def health():
    return {"status": "ok", "model": mcp_client.model}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(server_port))

    """
    # test command
    curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "deepseek-chat",
  "messages": [{"role":"user","content":"帮我查询一下数据库里的销售数据"}]
}'

    """

    # for test

    #text2sql
    #查询一下运动用品平均价格与食品平均价格哪个高
    #手工艺品总共有多少个商品
    #分析下梦彤的用户画像，从其购买的物品、职业、个人描述来分析
    #对用户“梦彤”做一下产品推荐
    
    # 相比上个月，这个月运动品价格涨的快还是食品价格涨的快

    # 

    #text2python
    #写一段从1+2+..+100的python代码，并执行
    #通过写一段python代码,计算一下365882*876545等于多少

    #text2sql+text2python实现chart+machine_learning
    #查询商品洗碗布的月销量数据，使用python语言生成代码完成绘制一张以月为维度的销量柱状图
    #查询商品"洗碗布"的月销量数据，绘制一张以月为维度的销量柱状图
    #查询银耳的用户评论和星级数据，并分析评论好坏与星级两者的相关性，是否是星级越高用户越满意
    #根据银耳前十二个月的销量，预测下一个月的销量，
    #根据银耳前十二个月的销量，预测下一个月的销量，并画出趋势图


