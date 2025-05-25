import base64
import os
import json
from PIL import Image, ImageDraw, ImageFont
from volcenginesdkarkruntime import Ark

# 建议将API Key设为环境变量或从配置文件读取，而不是硬编码
# ARK_API_KEY = os.environ.get("VOLC_ARK_API_KEY")
# 如果你在测试时仍想使用硬编码的密钥，可以取消下面一行的注释
ARK_API_KEY = "f65249de-2f94-4f9c-b654-8a4de76ad288"  # 你的API Key


# 初始化Ark客户端
# 最好在主程序中初始化一次，然后传递给函数，避免重复初始化
# 如果这个函数会被频繁调用，可以将client作为参数传入
# client = Ark(api_key=ARK_API_KEY)

def encode_image_to_base64(image_path):
    """将图片文件编码为Base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def detect_and_draw_watermarks(image_path: str, client: Ark) -> Image.Image:
    """
    检测图片中的水印并在图片上绘制边界框。

    Args:
        image_path (str): 输入图片的路径。
        client (Ark): 初始化好的Ark客户端实例。

    Returns:
        PIL.Image.Image: 绘制了水印边界框的图片对象。
                         如果未检测到水印或发生错误，则返回原始图片。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件未找到: {image_path}")

    # 1. 准备图片数据
    base64_image = encode_image_to_base64(image_path)

    # 从文件扩展名推断图片类型
    _, ext = os.path.splitext(image_path)
    image_type = ext.lower().replace('.', '')
    if image_type == 'jpg':
        image_type = 'jpeg'  # API可能期望jpeg

    data_uri = f"data:image/{image_type};base64,{base64_image}"

    # 2. 构建prompt (与原代码一致)
    prompt = f"""
请分析输入图像，检测所有水印（包括文字和图标）。

要求：
1. 仅返回一个 JSON 对象，禁止输出任何额外说明。
2. JSON 对象的键为水印标识：文字水印直接用检测到的文本（UTF-8，双引号包裹）；图标水印请生成唯一标识（如 "icon_watermark"，若有多个依次命名为 "icon_watermark_1"，"icon_watermark_2" 等）。
3. 每个键对应的值为一个浮点数组 [x_min, y_min, x_max, y_max]，表示水印框边界，坐标已根据图像尺寸归一化到 0–1，保留 6 位小数。
4. 无检测时返回 {{}}。

示例：
{{
  "Company © 2024": [0.123456, 0.234567, 0.345678, 0.456789],
  "icon_watermark": [0.500000, 0.600000, 0.700000, 0.800000]
}}
"""

    # 3. 调用API
    try:
        response = client.chat.completions.create(
            model="doubao-1-5-thinking-vision-pro-250428",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ],
                }
            ],
            extra_headers={'x-is-encrypted': 'true'},
            temperature=0,
            top_p=0.7,
            max_tokens=4096,
        )
        response_content = response.choices[0].message.content
    except Exception as e:
        print(f"调用API时发生错误: {e}")
        # 发生错误时，返回原始图片或抛出异常
        return Image.open(image_path)

    # 4. 解析JSON响应
    try:
        watermark_data = json.loads(response_content)
        if not isinstance(watermark_data, dict):
            print(f"API返回的不是有效的JSON对象: {response_content}")
            return Image.open(image_path)
    except json.JSONDecodeError:
        print(f"无法解析API返回的JSON: {response_content}")
        return Image.open(image_path)

    # 5. 加载图片并准备绘图
    img = Image.open(image_path).convert("RGB")  # 确保是RGB格式以便绘图
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    # (可选) 加载字体用于绘制标签
    try:
        # 尝试加载一个常用字体，你可能需要根据你的系统调整字体文件路径或名称
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font

    # 6. 绘制边界框
    if not watermark_data:
        print("未检测到水印。")
        return img  # 返回原始图片

    for label, normalized_coords in watermark_data.items():
        if not (isinstance(normalized_coords, list) and len(normalized_coords) == 4):
            print(f"水印 '{label}' 的坐标格式不正确: {normalized_coords}")
            continue

        nx_min, ny_min, nx_max, ny_max = normalized_coords

        # 还原坐标
        x_min = int(nx_min * img_width)
        y_min = int(ny_min * img_height)
        x_max = int(nx_max * img_width)
        y_max = int(ny_max * img_height)

        # 绘制矩形框
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

        # (可选) 绘制标签
        text_position = (x_min, y_min - 20 if y_min - 20 > 0 else y_min + 5)
        draw.text(text_position, label, fill="red", font=font)
        print(f"检测到水印: {label} at [{x_min}, {y_min}, {x_max}, {y_max}]")

    return img


# --- 主程序示例 ---
if __name__ == "__main__":
    # 确保你有一个名为 "a3.png" 的图片在脚本同目录下，或者修改为你的图片路径
    # 你也可以使用其他的图片，如 "test_image_with_watermark.jpg"
    input_image_path = "a3.png"
    output_image_path = "a3_watermarked_output.png"

    if not ARK_API_KEY or ARK_API_KEY == "YOUR_API_KEY":
        print("请设置您的火山引擎ARK API Key (VOLC_ARK_API_KEY)")
    else:
        # 在这里初始化客户端
        ark_client = Ark(api_key=ARK_API_KEY)

        try:
            # 调用函数
            processed_image = detect_and_draw_watermarks(input_image_path, ark_client)

            # 显示或保存图片
            processed_image.show()  # 显示图片
            processed_image.save(output_image_path)  # 保存图片
            print(f"处理后的图片已保存到: {output_image_path}")

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"处理图片时发生意外错误: {e}")
