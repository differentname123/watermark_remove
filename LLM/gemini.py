import os
import time

# 假设 common_utils.common_utils 存在且 get_config 函数可用
from common_utils.common_utils import get_config

# 设置代理环境变量，这部分通常放在程序启动时
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 使用 google.genai 风格的导入
# To run this code you need to install the following dependencies:
# pip install google-genai
import google.genai
from google.genai import types # 需要导入 types 来构建 contents 和 config

# 获取 API 密钥 (保留原代码逻辑)
API_KEY = get_config("gemini_api_key")
print("正在使用 Gemini API 密钥:", API_KEY)

# 修改 get_llm_content 函数
def get_llm_content(API_KEY=get_config("gemini_api_key"),
                    prompt='你好，Gemini！请介绍一下你自己。',
                    model_name="gemini-2.5-flash-preview-04-17"):
    """
    使用 google.genai.Client 调用 Gemini 模型生成文本内容。
    如果首次生成内容失败，则尝试使用备用模型 gemini-2.5-flash-lite-preview-06-17。

    Args:
        API_KEY: Gemini API 密钥。
        prompt: 输入给模型的文本提示。
        model_name: 要使用的 Gemini 模型名称，默认使用 gemini-2.5-flash-preview-04-17。

    Returns:
        生成的文本内容字符串，如果发生错误则返回 None。
    """
    try:
        # 1. 创建 google.genai.Client 实例，使用传入的 API_KEY
        client = google.genai.Client(api_key=API_KEY)

        # 2. 构建 contents 列表，使用 types.Content 和 types.Part
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        # 3. 构建生成文本配置
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=24576,
            ),
            response_mime_type="text/plain",
        )

        # 尝试首次调用指定的模型生成内容
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )
        generated_text = response.text

        # 如果首次生成内容失败，则使用备用模型尝试
        if not generated_text:
            print("首次生成内容失败，尝试使用备用模型 gemini-2.5-flash-lite-preview-06-17...")
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite-preview-06-17",
                contents=contents,
                config=generate_content_config,
            )
            generated_text = response.text

        return generated_text

    except Exception as e:
        print(f"发生错误: {e}")
        print("请检查您的 API 密钥是否正确，网络连接是否正常，以及代理设置是否工作。")
        print("同时，请确保您已正确安装 google-genai 库 (pip install -q -U google-genai)。")
        return None

# 测试函数 (保留原代码逻辑，但调用新的函数并处理返回值)
if __name__ == "__main__":
    print("--- Testing get_llm_content ---")
    # 使用之前获取的 API_KEY 进行测试调用
    start_time = time.time()
    result = get_llm_content(
        API_KEY=API_KEY,
        prompt='你好，Gemini！请介绍一下你自己。',
        model_name="gemini-2.5-flash-preview-04-17"
    )

    if result:
        print("\n--- Generated Content ---")
        print(result)
    else:
        print("\n--- Generation Failed ---")
    print("执行时间: {:.2f} 秒".format(time.time() - start_time))