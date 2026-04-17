from openai import OpenAI
import os

def test_qwen_model():
    # 若没有配置环境变量，可用百炼API Key将下行替换为：api_key="sk-xxx"。但不建议在生产环境中直接将API Key硬编码到代码中，以减少API Key泄露风险。
    # api_key=os.getenv("DASHSCOPE_API_KEY")
    api_key = "sk-89d86e2a28be4e418355db87d20f3b5e"

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-turbo",  # 通义千问模型
        messages=[{"role": "user", "content": "你是谁？"}]
    )

    print(completion.choices[0].message.content)

if __name__=="__main__":
    test_qwen_model()
    pass