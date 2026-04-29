import asyncio, os
import time
import json 
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from contextlib import asynccontextmanager
from mcp import ClientSession
from contextlib import AsyncExitStack
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

#加载MCP服务器的地址
server_url = os.getenv("server_url")

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
    
    async def connect(self):
        # 启动时执行（等价于 startup）
        await self.connect_to_server(
            "search_db_mcp",
            f"http://{server_url}:9004/mcp",
            "streamable-http"
        )
        await self.connect_to_server(
            "python_chart_mcp",
            f"http://{server_url}:9012/mcp",
            "streamable-http"
        )
        await self.connect_to_server(
            "machine_learning_mcp",
            f"http://{server_url}:9003/mcp",
            "streamable-http"
        )

    async def process_query(self, tool_name, input) -> dict:
        """
        接收工具名称和输入，返回工具执行的结果
        返回: 
        """
        print(f"\n======================================")


        tools = await self.list_all_tools()
        print(f"🔧 加载可用工具数量: {len(tools)}")
        if tools:
            print(f"📋 工具列表: {json.dumps(tools, ensure_ascii=False, indent=2)}")

        # =============================================================================



        args = input

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
        print(content)

            
    async def clean(self):
        """清理所有资源"""
        self.exit_stack.aclose()
        self.sessions.clear()
        self.tools_map.clear()



async def main():
    # 启动并初始化 MCP 客户端
    client = MCPClient()
    print(server_url)
    try:
        # 连接多个 MCP Streamable HTTP 服务器
        #连接数据库数据分析MCP服务器
        await client.connect()
        # --------------------call mcp service--------------------

        await client.process_query(
            "run_python_script_tool", 
            {
                "script_content": "import matplotlib.pyplot as plt\nimport numpy as np\n\n# 数据\ncategories = ['运动用品', '食品']\navg_prices = [297.74, 5.92]\n\n# 创建图表\nfig, ax = plt.subplots(figsize=(10, 6))\n\n# 创建柱状图\nbars = ax.bar(categories, avg_prices, color=['#FF6B6B', '#4ECDC4'], width=0.6)\n\n# 添加数值标签\nfor bar, price in zip(bars, avg_prices):\n    height = bar.get_height()\n    ax.text(bar.get_x() + bar.get_width()/2., height + 5,\n            f'¥{price:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')\n\n# 设置标题和标签\nax.set_title('运动用品 vs 食品 平均价格对比', fontsize=16, fontweight='bold', pad=20)\nax.set_ylabel('平均价格（元）', fontsize=12)\nax.set_xlabel('商品类别', fontsize=12)\n\n# 设置y轴范围\nax.set_ylim(0, max(avg_prices) * 1.2)\n\n# 添加网格\nax.grid(axis='y', alpha=0.3, linestyle='--')\n\n# 添加价格差异说明\nprice_ratio = avg_prices[0] / avg_prices[1]\nax.text(0.5, max(avg_prices) * 0.9, \n        f'运动用品平均价格是食品的{price_ratio:.0f}倍', \n        ha='center', fontsize=12, fontweight='bold',\n        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))\n\nplt.tight_layout()"
            }
        )


    finally:
        # 清理资源
        await client.clean()


    
if __name__ == "__main__":
    asyncio.run(main())