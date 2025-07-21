import os
import time
import base64
# 假设 common_utils 是您本地的模块
from common_utils.common_utils import get_config
from PIL import Image

# 新增：用于识别图片文件类型
import mimetypes

# 设置代理环境变量
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import google.genai as genai
from google.genai import types

# 获取 API 密钥
API_KEY = get_config("gemini_api_key")
print(f"[INFO] 正在使用 Gemini API 密钥: {API_KEY}")

# === 新增：视频内容分析函数（使用 google.generativeai） ===
# 注意：您原代码中此处导入了两次 google.generativeai，一次为 genai，一次为 genai_flash，我将保持原样。
import google.generativeai as genai_flash
from google.api_core import exceptions as ga_exceptions


def get_llm_content_gemini_flash_video(
        api_key: str = API_KEY,
        prompt: str = '视频中的内容是什么',
        video_path: str = 'test.mp4'
) -> str:
    """
    使用 Gemini 1.5 Flash 模型分析视频内容并返回文本描述。
    """
    try:
        # 配置 API
        genai_flash.configure(api_key=api_key)

        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            return f"错误: 视频文件未找到 -> {video_path}"

        # 上传并等待处理
        video_file = genai_flash.upload_file(path=video_path)
        while video_file.state.name == "PROCESSING":
            print("等待视频文件处理完成...")
            time.sleep(10)
            video_file = genai_flash.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            return f"错误: 视频文件 '{video_path}' 处理失败。"

        # 调用 Gemini 多模态模型
        # 注意：您原代码中此处模型名为 gemini-2.5-pro，我将保持原样。
        model = genai_flash.GenerativeModel(model_name="gemini-2.5-pro")
        response = model.generate_content(
            [video_file, prompt],
            request_options={"timeout": 600}
        )
        return response.text

    except ga_exceptions.GoogleAPICallError as e:
        return f"调用 Gemini API 时发生网络或权限错误: {e}"
    except Exception as e:
        return f"处理过程中发生未知错误: {e}"


def get_llm_content_gemini2flash(api_key: str = API_KEY,
                                 prompt: str = '你好，Gemini！请介绍一下你自己。') -> str:
    """
    使用 gemini-2.0-flash 模型生成内容
    """
    print("[INFO] 使用模型: gemini-2.0-flash")
    client = genai.Client(api_key=api_key)

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]
    config = types.GenerateContentConfig(response_mime_type="text/plain")

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=config,
    )
    return response.text


def get_llm_content_sub(api_key: str = API_KEY,
                        prompt: str = '你好，Gemini！请介绍一下你自己。',
                        model_name: str = "gemini-2.5-flash-preview-04-17") -> str:
    """
    使用指定 Gemini 模型生成内容
    """
    print(f"[INFO] 使用模型: {model_name}")
    client = genai.Client(api_key=api_key)

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=24576),
        response_mime_type="text/plain"
    )

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )
    return response.text


def get_llm_content(api_key: str = API_KEY,
                    prompt: str = '你好，Gemini！请介绍一下你自己。') -> str | None:
    """
    优先尝试主模型，若失败则依次使用备用模型生成内容。
    """
    try:
        try:
            return get_llm_content_sub(api_key, prompt, "gemini-2.5-pro")
        except Exception as e1:
            print(f"[WARN] 主模型失败: {e1}")
            try:
                return get_llm_content_sub(api_key, prompt, "gemini-2.5-flash-preview-04-17")
            except Exception as e2:
                print(f"[WARN] 备用模型失败: {e2}")
                return get_llm_content_gemini2flash(api_key, prompt)

    except Exception as e:
        print(f"[ERROR] 内容生成失败: {e}")
        print("[TIPS] 请检查以下内容：")
        print(" - API 密钥是否正确")
        print(" - 网络连接及代理设置")
        print(" - 是否安装了 `google-genai`（pip install -q -U google-genai）")
        return None


# ==============================================================================
# ===               ↓↓↓  以下是本次新增的代码  ↓↓↓               ===
# ==============================================================================

def analyze_images_gemini(api_key=API_KEY,prompt='每张图片的内容是什么', image_paths=['a.jpg']) -> str:
    """
    使用 Gemini Vision 模型分析一个或多个图片内容并返回文本描述。
    函数风格与 get_llm_content_gemini_flash_video 保持一致。

    Args:
        api_key (str): 您的 Google AI API 密钥。
        prompt (str): 指导模型如何分析图片的文本提示。
        image_paths (list[str]): 包含待分析图片文件路径的列表。

    Returns:
        str: 模型生成的文本描述。如果发生错误，则返回错误信息字符串。
    """
    try:
        # 1. 配置 API
        genai_flash.configure(api_key=api_key)

        # 2. 准备模型输入内容 (prompt + images)
        prompt_parts = [
            prompt,
            "下面我将以'文件名:'的格式，在每个图片前提供其名称，请据此作答。"
        ]
        for path in image_paths:
            if not os.path.exists(path):
                return f"错误: 图片文件未找到 -> {path}"

            filename_identifier = f"{os.path.basename(path)}:"
            prompt_parts.append(filename_identifier)

            # 使用 PIL 打开图片，这是 genai_flash 模式下最兼容的方式
            img = Image.open(path)
            prompt_parts.append(img)

        # 3. 调用 Gemini 多模态模型
        model = genai_flash.GenerativeModel(model_name="gemini-2.5-pro")
        response = model.generate_content(
            prompt_parts,
            request_options={"timeout": 600}
        )
        return response.text

    except ga_exceptions.GoogleAPICallError as e:
        return f"调用 Gemini API 时发生网络或权限错误: {e}"
    except Exception as e:
        # 提供更详细的错误信息
        return f"处理过程中发生未知错误: {e.__class__.__name__}: {e}"

if __name__ == "__main__":
    print("[TEST] 正在测试 get_llm_content")
    start_time = time.time()

    result = get_llm_content()
    if result:
        print("\n[RESULT] 模型输出：\n")
        print(result)
    else:
        print("\n[FAIL] 内容生成失败")

    print(f"[INFO] 执行时间: {time.time() - start_time:.2f} 秒")

    result = get_llm_content_gemini_flash_video(
        prompt="请详细描述这个视频的内容，分点说明。",
        video_path="test.mp4"
    )
    print(result)


    image_analysis_result = analyze_images_gemini(
        image_paths=['a.jpg','a3.png']
    )

    print("\n[RESULT] 图片分析模型输出：\n")
    print(image_analysis_result)