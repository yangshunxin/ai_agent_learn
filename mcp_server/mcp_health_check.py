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