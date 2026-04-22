import os
from dashscope import Generation
from openai import OpenAI
import dashscope
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
from dotenv import load_dotenv
#加载.env文件中的环境变量
load_dotenv()

def LLM_replay(message):
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "你是谁？"},
    # ]
    response = Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
        api_key=os.getenv("QWEN_API_KEY"),
        model="qwen-turbo", # qwen3-max
        messages=[{"role": "user", "content": message}],
        result_format="message",
        # 开启深度思考
        enable_thinking=False,
        stream=False
    )

    if response.status_code == 200:
        # 打印思考过程
        if "reasoning_content" in response.output.choices[0].message:
            print("=" * 20 + "思考过程" + "=" * 20)
            print(response.output.choices[0].message.reasoning_content)
        
        # 打印回复
        print("=" * 20 + "完整回复" + "=" * 20)
        print(response.output.choices[0].message.content)
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")

def LLM_OpenAI(message):
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
    )

    response = client.responses.create(
        model="qwen3-max",
        input=[{"role": "user", "content": message}],
        extra_body={
            "enable_thinking": False  # 启用思考模式
        }
    )

    # 遍历输出项
    for item in response.output:
        if item.type == "reasoning":
            # 打印推理过程摘要
            print("【推理过程】")
            for summary in item.summary:
                print(summary.text[:500])  # 截取前500字符
            print()
        elif item.type == "message":
            # 打印最终答案
            print("【最终答案】")
            print(item.content[0].text)

if __name__=="__main__":
    LLM_replay("你是谁？")
    print("==="*20)
    LLM_OpenAI("你是谁？")
