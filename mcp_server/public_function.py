from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from dashscope import Generation
import dashscope
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
import asyncio
#加载.env文件中的环境变量
load_dotenv()
#定义大模型调用函数，用于处理文本类模型生成功能
async def LLM_replay_old(messages):
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


def LLM_replay(messages):
    """
    prompt_template:大模型调用的提示词模板
    message:大模型调用的用户输入
    """
    response = Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen3-max",
        messages=[{"role":"user","content":messages}],
        result_format="message",
        # 开启深度思考
        enable_thinking=False,
        temperature = 0
    )

    if response.status_code == 200:
        # # 打印思考过程
        # print("=" * 20 + "思考过程" + "=" * 20)
        # print(response.output.choices[0].message.reasoning_content)
        
        # # 打印回复
        # print("=" * 20 + "完整回复" + "=" * 20)
        # print(response.output.choices[0].message.content)
        return response.output.choices[0].message.content
    else:
        # print(f"HTTP返回码：{response.status_code}")
        # print(f"错误码：{response.code}")
        # print(f"错误信息：{response.message}")
        return "call llm error"

# result=await LLM_replay(messages="你好")
# print(result)

