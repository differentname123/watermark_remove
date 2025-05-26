# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/5/26 18:52
:last_date:
    2025/5/26 18:52
:description:
    
"""
import subprocess
import time

from common_utils.image_utils import compress_jpeg_with_pillow


def enhance_image(
    input_path,
    output_path,
    model_name="realesrgan-x4plus",
    exe_path="realesrgan-ncnn-vulkan"
):
    """
    使用 realesrgan-ncnn-vulkan 进行图像增强

    参数:
        input_path (str): 输入图片路径
        output_path (str): 输出图片路径
        model_name (str): 模型名称（默认 'realesrgan-x4plus'）
        exe_path (str): 可执行文件路径（默认 'realesrgan-ncnn-vulkan'，假设已在环境变量中）

    返回:
        int: 命令的返回码，0 表示成功
    """
    start_time = time.time()
    cmd = [
        exe_path,
        "-i", input_path,
        "-o", output_path,
        "-n", model_name
    ]
    try:
        result = subprocess.run(cmd, check=True)
        print(f"图像增强完成：{output_path}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print("增强图像时出错：", e)
        return e.returncode
    print(f"增强图像耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    # 示例用法
    input_image = "a.jpg"  # 输入图片路径
    output_image = "enhance_a.jpg"  # 输出图片路径
    enhance_image(input_image, output_image)
    compress_jpeg_with_pillow(output_image, output_image)