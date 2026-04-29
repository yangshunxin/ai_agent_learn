import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# 改成你的 MCP 服务器地址
SERVER_HOST = "121.34.54.32"

MCP_SERVER_CONFIG = {
    "search_db_mcp": {
        "url": f"http://{SERVER_HOST}:9004/sse",
        "transport": "sse",
        "timeout": 30000,
    },
    "machine_learning_mcp": {
        "url": f"http://{SERVER_HOST}:9003/mcp",
        "transport": "streamable_http",
        "timeout": 30000,
    },
    "python_chart_mcp": {
        "url": f"http://{SERVER_HOST}:9002/mcp",
        "transport": "streamable_http",
        "timeout": 30000,
    },
    "ywfl_mcp": {
        "url": f"http://{SERVER_HOST}:9005/mcp",
        "transport": "streamable_http",
        "timeout": 30000,
    },
}
class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions = {}
        self.tools_map = {}


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


async def check_mcp_server(server_name: str, config: dict):
    """测试单个 MCP 服务是否能连接、能获取工具"""
    print(f"\n===== 正在测试 {server_name} =====")
    try:
        client = MultiServerMCPClient({server_name: config})
        async with client.session(server_name) as session:
            tools = await load_mcp_tools(session)
            print(f"✅ {server_name} 连接成功！")
            print(f"可用工具：{[t.name for t in tools]}")
        return True
    except Exception as e:
        print(f"❌ {server_name} 失败：{repr(e)}")
        return False


async def main():
    results = {}
    for name, cfg in MCP_SERVER_CONFIG.items():
        ok = await check_mcp_server(name, cfg)
        results[name] = ok

    print("\n===== 最终检查结果 =====")
    for name, ok in results.items():
        status = "✅ 正常" if ok else "❌ 异常"
        print(f"{name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())