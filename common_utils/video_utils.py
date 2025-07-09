# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/7/8 17:52
:last_date:
    2025/7/8 17:52
:description:
    
"""

import os
import subprocess
import json


def probe_video(path):
    """用 ffprobe 返回字典：{width, height, fps}"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        path
    ]
    # 注意：ffprobe 已经使用了 -v error，所以它本身就很安静，无需修改。
    out = subprocess.check_output(cmd)
    info = json.loads(out)["streams"][0]
    num, den = map(int, info["r_frame_rate"].split("/"))
    return {
        "width": info["width"],
        "height": info["height"],
        "fps": num / den
    }


def probe_duration(path):
    """返回视频时长（秒）"""
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ])
    return float(out)

def add_image_to_video_end(
    video_path: str,
    image_path: str,
    output_path: str,
    image_duration: float = 1.0,
    max_retries: int = 3
) -> None:
    """
    将一张图片拼接到视频末尾。如果输出文件小于输入视频文件，最多重试 max_retries 次。

    :param video_path: 输入视频路径
    :param image_path: 输入图片路径
    :param output_path: 输出视频路径
    :param image_duration: 图片在视频末尾的持续时长（秒）
    :param max_retries: 最大重试次数
    :raises RuntimeError: 如果所有重试均失败
    """
    # 初次探测视频元信息、时长
    meta = probe_video(video_path)
    video_dur = probe_duration(video_path)
    total_dur = video_dur + image_duration

    # 构造 filter_complex 字符串
    filter_complex = (
        f"[1:v]fps={meta['fps']:.2f},"
        f"scale={meta['width']}:{meta['height']},"
        "format=yuv420p[img];"
        "[0:v][img]concat=n=2:v=1:a=0[v]"
    )

    # 公共 ffmpeg 参数
    base_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-loop", "1", "-t", str(image_duration), "-i", image_path,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",            # 如果有音频就映射
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",             # 必须重新编码才能用 apad
        "-af", "apad",             # 自动补零
        "-t", str(total_dur),      # 强制输出时长
    ]

    input_size = os.path.getsize(video_path)

    for attempt in range(1, max_retries + 1):
        try:
            # 每次都重新生成输出文件
            cmd = base_cmd + [output_path]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            # ffmpeg 出错，记录并重试
            print(f"[警告] 第 {attempt} 次合成失败：{e}")
        else:
            # 检查输出文件大小
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path)
                if output_size >= input_size:
                    # 成功且大小正常，退出循环
                    return
                else:
                    print(f"[警告] 第 {attempt} 次生成的视频大小 ({output_size}) 小于输入视频大小 ({input_size})，重试中...")
            else:
                print(f"[警告] 第 {attempt} 次没有生成输出文件，重试中...")

    # 如果循环结束仍未成功，则抛出异常
    raise RuntimeError(f"多次尝试后仍未生成有效视频（{max_retries} 次）")


def cover_video_area(
    video_path: str,
    output_path: str,
    top_left,
    bottom_right,
    color: str = 'black'
) -> None:
    # ... (你之前的硬遮挡函数保持不变) ...
    x1, y1 = top_left
    x2, y2 = bottom_right
    if not (x2 > x1 and y2 > y1):
        raise ValueError("右下角坐标必须大于左上角坐标")
    width = x2 - x1
    height = y2 - y1
    vf_filter = f"drawbox=x={x1}:y={y1}:w={width}:h={height}:color={color}:t=fill"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-vf", vf_filter, "-c:a", "copy", output_path]
    try:
        subprocess.run(cmd, check=True)
        print(f"成功！已将带硬遮挡的视频保存至: {output_path}")
    except FileNotFoundError:
        print("[错误] ffmpeg 未安装或未在系统 PATH 中。请先安装 ffmpeg。")
        raise
    except subprocess.CalledProcessError as e:
        print(f"[错误] ffmpeg 执行失败。返回码: {e.returncode}")
        raise


# ==============================================================================
# ======================   新增的“温和遮挡”函数   =======================
# ==============================================================================

def cover_video_area_gently(
    video_path: str,
    output_path: str,
    top_left,
    bottom_right,
    mode = 'blur',
    strength: int = 50
) -> None:
    """
    使用 ffmpeg 以更温和的方式（模糊或马赛克）遮挡视频的指定区域。

    :param video_path: 输入视频的路径。
    :param output_path: 输出视频的路径。
    :param top_left: 遮挡区域左上角的坐标 (x1, y1)。
    :param bottom_right: 遮挡区域右下角的坐标 (x2, y2)。
    :param mode: 遮挡模式, 'blur' (模糊) 或 'pixelate' (马赛克)。
    :param strength: 效果强度。
                     对于 'blur'，值越大越模糊 (建议范围 10-100)。
                     对于 'pixelate'，值越大马赛克格子越小 (建议范围 20-100, 表示格子宽度像素)。
    :raises ValueError: 如果坐标或模式无效。
    :raises subprocess.CalledProcessError: 如果 ffmpeg 命令执行失败。
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    if not (x2 > x1 and y2 > y1):
        raise ValueError("右下角坐标必须大于左上角坐标")

    width = x2 - x1
    height = y2 - y1

    print(f"准备 '{mode}' 遮挡区域：位置=({x1}, {y1}), 尺寸={width}x{height}, 强度={strength}")

    # 构建 filter_complex 字符串
    if mode == 'blur':
        # -- [修改点] --
        # 明确指定 luma_radius (lr)，这是主要模糊参数。
        # 我们将 strength 直接赋给它。
        # 对于 chroma_radius (cr)，我们给一个较小且安全的值，或者让它与亮度半径成比例但不超过其限制。
        # 这里我们简单地让色度模糊等于亮度模糊，但限制其最大值为 27（根据报错信息）。
        chroma_strength = min(strength, 27)
        effect_filter = f"boxblur=luma_radius={strength}:lr={strength}:chroma_radius={chroma_strength}:cr={chroma_strength}"
        # `lr` 是 `luma_radius` 的简写，`cr` 是 `chroma_radius` 的简写。
        # 同时提供全名和简写是为了兼容不同版本的ffmpeg，更稳妥。
        # 更简洁的写法是：f"boxblur=lr={strength}:cr={min(strength, 27)}"

    elif mode == 'pixelate':
        # pixelize滤镜通常没有这个问题，保持原样
        effect_filter = f"pixelize={strength}"
    else:
        raise ValueError(f"不支持的模式: '{mode}'。请选择 'blur' 或 'pixelate'。")

    # [0:v] 是主视频流
    # split=2 将主视频流复制成两份，我们命名为 [main] 和 [cropped]
    # [cropped] 流被 crop 滤镜处理，裁剪出目标区域
    # 裁剪后的流应用 effect_filter (模糊或马赛克)
    # 最后，[main] 流作为背景，用 overlay 滤镜将处理后的流覆盖到指定位置
    filter_complex = (
        f"[0:v]split=2[main][cropped];"
        f"[cropped]crop={width}:{height}:{x1}:{y1},"
        f"{effect_filter}[effect];"
        f"[main][effect]overlay={x1}:{y1}"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-c:a", "copy",
        output_path
    ]

    try:
        print(f"正在执行 ffmpeg 命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"成功！已将带 '{mode}' 遮挡的视频保存至: {output_path}")
    except FileNotFoundError:
        print("[错误] ffmpeg 未安装或未在系统 PATH 中。请先安装 ffmpeg。")
        raise
    except subprocess.CalledProcessError as e:
        print(f"[错误] ffmpeg 执行失败。返回码: {e.returncode}")
        print(f"命令: {' '.join(e.cmd)}")
        # 尝试解码 stderr 以获取更清晰的错误信息
        stderr_output = ""
        if e.stderr:
            try:
                stderr_output = e.stderr.decode('utf-8', errors='ignore')
            except Exception:
                stderr_output = repr(e.stderr)
        print(f"FFMPEG 错误输出:\n{stderr_output}")
        raise


if __name__ == '__main__':
    video_path = "test.mp4"
    output_path = "output_video.mp4"
    top_left =  (8, 499)
    bottom_right = (1271, 718)
    color = "black"

    cover_video_area_gently(
        video_path=video_path,
        output_path=output_path,
        top_left=top_left,
        bottom_right=bottom_right,
        mode='blur',
        strength=50  # 模糊强度，可以调整
    )