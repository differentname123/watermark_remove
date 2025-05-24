
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
import google.generativeai as genai
API_KEY = "AIzaSyCpV0OZ34nuyxVO1bfYgSmVMphMAno6RxQ"  # 请替换为您的 API 密钥
try:
    # 配置 API 密钥
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

    # 定义您想要发送给模型的提示（prompt）
    prompt = "你好，Gemini！请介绍一下你自己。"

    # 调用 API 生成内容
    response = model.generate_content(prompt)

    # 打印生成的文本内容
    print(response.text)

except Exception as e:
    print(f"发生错误: {e}")
    print("请检查您的 API 密钥是否正确，以及网络连接是否正常。")
    print("同时，请确保您已正确安装 google-generativeai 库 (pip install -q -U google-generativeai)。")