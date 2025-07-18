# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/7/8 17:52
:last_date:
    2025/7/9 10:30
:description:
    包含视频处理（添加图片、遮挡、添加字幕）的工具函数集
"""

import os
import random
import shutil
import subprocess
import json
import tempfile

from PIL import ImageFont

from common_utils.common_utils import time_to_ms


def cut_audio_segment(input_audio_path: str, start_time: float, end_time: float, output_audio_path: str):
    """
    从音频中截取指定时间段，保存为新的音频文件。

    参数说明：
    - input_audio_path: 输入音频文件路径
    - start_time: 开始时间（单位：秒，可为小数）
    - end_time: 结束时间（单位：秒）
    - output_audio_path: 输出音频文件路径
    """
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError("end_time 必须大于 start_time")

    cmd = [
        "ffmpeg", "-y",           # <--- 添加自动覆盖参数
        "-v", "error",
        "-ss", str(start_time),
        "-i", input_audio_path,
        "-t", str(duration),
        "-c", "copy",             # 快速截取，不转码
        output_audio_path
    ]
    subprocess.run(cmd, check=True)

def extract_audio_from_video(video_path: str, audio_output_path: str):
    """
    从视频中提取音频为 wav 格式，适用于 Demucs 分离。
    """
    cmd = [
        "ffmpeg", "-y",          # <--- 自动覆盖已存在文件
        "-v", "error",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        audio_output_path
    ]
    subprocess.run(cmd, check=True)

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
        "-map", "0:a?",  # 如果有音频就映射
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",  # 必须重新编码才能用 apad
        "-af", "apad",  # 自动补零
        "-t", str(total_dur),  # 强制输出时长
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
                if output_size >= input_size * 0.8:
                    # 成功且大小正常，退出循环
                    return
                else:
                    print(
                        f"[警告] 第 {attempt} 次生成的视频大小 ({output_size}) 小于输入视频大小 ({input_size * 0.8})，重试中...")
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
    # ... (硬遮挡函数保持不变) ...
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


import shlex
import subprocess
import json
import re


# 假设 _get_video_resolution 已经定义好了，这里提供一个实现
def _get_video_resolution(video_path: str) -> tuple[int, int]:
    """使用 ffprobe 获取视频分辨率"""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", video_path
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
        width = data['streams'][0]['width']
        height = data['streams'][0]['height']
        return width, height
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError, IndexError) as e:
        raise RuntimeError(f"无法获取视频分辨率: {e}")


def cover_video_area_gently(
        video_path: str,
        output_path: str,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        mode: str = 'blur',
        strength: int = 50
):
    """
    在 video_path 指定的视频上，对 [top_left, bottom_right] 区域做遮挡，
    并输出到 output_path。支持 'blur'、'gblur'（更灵活的 Gaussian Blur）和 'pixelate' 三种模式。
    strength 参数：
      - blur 模式：控制 luma_radius（及 gblur 的 sigma）
      - pixelate 模式：控制方格大小
    """
    # 1. 检查坐标合法性
    x1, y1 = top_left
    x2, y2 = bottom_right
    if not (isinstance(x1, int) and isinstance(y1, int)
            and isinstance(x2, int) and isinstance(y2, int)):
        raise ValueError("坐标必须为整数元组 (x, y)")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("右下角坐标必须大于左上角坐标")

    # 2. 获取视频分辨率并校验区域不超出边界
    vid_w, vid_h = _get_video_resolution(video_path)
    if not (0 <= x1 < vid_w and 0 <= x2 <= vid_w
            and 0 <= y1 < vid_h and 0 <= y2 <= vid_h):
        raise ValueError(f"裁剪区域超出视频范围 ({vid_w}x{vid_h})")

    width = x2 - x1
    height = y2 - y1

    print(f"[INFO] 输入视频分辨率：{vid_w}x{vid_h}")
    print(f"[INFO] 遮挡区域：位置=({x1},{y1}), 大小={width}x{height}, 模式={mode}, 强度={strength}")

    # 3. 构造滤镜
    if mode == 'blur':
        # 根据 FFmpeg 错误日志，boxblur 的半径有上限（此环境下为 23）。
        # 我们需要将 strength 限制在这个范围内。
        # gblur 的 sigma 则没有这个硬性限制，因此更灵活。
        BLUR_RADIUS_MAX = 23

        # 限制 luma 强度
        luma_strength = min(strength, BLUR_RADIUS_MAX)

        # 限制 chroma 强度，它同时受限于 strength、动态上限和滤镜本身的最大值
        max_chroma_dynamic = height // 4
        chroma_strength = min(strength, max_chroma_dynamic, BLUR_RADIUS_MAX)

        effect = f"boxblur=luma_radius={luma_strength}:lr={luma_strength}" \
                 f":chroma_radius={chroma_strength}:cr={chroma_strength}"

    elif mode == 'gblur':
        # Gaussian blur（没有色度半径限制）
        effect = f"gblur=sigma={strength}"
    elif mode == 'pixelate':
        effect = f"pixelize={strength}"
    else:
        raise ValueError("不支持的模式，请选择 'blur'、'gblur' 或 'pixelate'")

    # 4. 完整 filter_complex
    filter_complex = (
        # split 主流和裁剪流
        f"[0:v]split=2[main][crop];"
        # 裁剪 + 效果
        f"[crop]crop={width}:{height}:{x1}:{y1},{effect}[eff];"
        # 合成
        f"[main][eff]overlay={x1}:{y1}"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-c:a", "copy",
        output_path
    ]

    # 5. 执行并捕获任何错误
    try:
        print(f"[INFO] 运行命令：{' '.join(shlex.quote(c) for c in cmd)}")
        # 注意：在Windows上，text=True可能会导致编码问题，如果stderr出现乱码，可以尝试指定encoding
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg 错误（{proc.returncode}）：\n{proc.stderr}")
        print(f"[SUCCESS] 已生成：{output_path}")
    except FileNotFoundError:
        raise FileNotFoundError("未检测到 ffmpeg，请先安装并添加到 PATH。")
    return vid_w, vid_h

# ==============================================================================
# ========================   新增的“添加字幕”函数   =========================
# ==============================================================================

def _parse_subtitle_time(time_str: str) -> float:
    """
    将各种格式的时间字符串统一转换为秒 (float)。
    这个函数现在是 time_to_ms 的一个包装器，以保证健壮性和一致性。
    """
    # 1. 调用我们已经写好的、非常健壮的 time_to_ms 函数
    milliseconds = time_to_ms(time_str)

    # 2. 将毫秒转换为秒 (float)，以匹配原始函数的返回值类型
    return milliseconds / 1000.0

def _escape_ffmpeg_text(text: str) -> str:
    """
    为ffmpeg的drawtext滤镜转义特殊字符。
    """
    # 转义 \ ' % :
    # 单引号 ' 替换为视觉上相似的 ’，避免破坏滤镜语法
    return text.replace('\\', '\\\\').replace("'", "’").replace('%', r'\%').replace(':', r'\:')


# [新增] 辅助函数：获取视频的宽度和高度
def get_video_dimensions(video_path: str) -> (int, int):
    """
    使用 ffprobe 获取视频的宽度和高度。
    此版本确保 video_path 被正确传递，并提供详细的错误处理。
    """
    # 检查文件是否存在，提前给出更友好的提示
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件未找到，请检查路径: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path  # <-- 修正：将 video_path 作为命令的一部分
    ]

    try:
        # 使用 check=True, ffprobe 失败时会抛出 CalledProcessError
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        data = json.loads(result.stdout)

        # 健壮性检查：确保返回的 JSON 结构符合预期
        if "streams" in data and len(data["streams"]) > 0:
            stream = data["streams"][0]
            if "width" in stream and "height" in stream:
                return stream["width"], stream["height"]

        # 如果 JSON 结构不符
        raise ValueError("在 ffprobe 的输出中未找到有效的视频流信息。")

    except FileNotFoundError:
        # 如果 ffprobe 命令本身就找不到
        print("错误: ffprobe 命令未找到。请确保 ffmpeg (及 ffprobe) 已安装并在系统 PATH 中。")
        raise
    except subprocess.CalledProcessError as e:
        # 捕获 ffprobe 执行失败的错误，并打印其 stderr
        print(f"ffprobe 执行失败。返回码: {e.returncode}")
        # ffprobe 的错误信息通常在 stderr 中，这对于调试至关重要
        print(f"ffprobe 的原始错误输出:\n---\n{e.stderr.strip()}\n---")
        raise ValueError(f"无法从视频 {video_path} 中解析出尺寸。")
    except json.JSONDecodeError:
        # 如果 ffprobe 输出的不是有效的 json
        print(f"ffprobe 输出了非预期的内容，无法解析为JSON。输出内容: {result.stdout}")
        raise ValueError(f"无法解析来自 ffprobe 的视频尺寸信息。")

def _format_time_for_ffmpeg(seconds: float) -> str:
    # 辅助函数：将秒数格式化回 FFmpeg 需要的格式
    return f"{seconds:.3f}"

# [新增] 辅助函数：处理并分割过长的字幕
def _process_and_split_subtitles(
        subtitles_info,
        font: ImageFont.FreeTypeFont,
        max_width: int
):
    """
    预处理字幕列表，将过长的字幕分割成多段，直到每段都不超过最大宽度 max_width。
    """
    processed_subs = []
    split_chars = ['，', '。', '？', '！', '；', ',', '.', '?', '!', ';']

    for sub in subtitles_info:
        # 队列：存放 (startTime_str, endTime_str, text) 待处理
        segments_to_process = [(sub['startTime'], sub['endTime'], sub['optimizedText'])]

        while segments_to_process:
            start_str, end_str, text = segments_to_process.pop(0)
            # 计算渲染宽度
            try:
                text_width = font.getlength(text)
            except AttributeError:
                text_width = font.getsize(text)[0]

            # 如果宽度合格，直接输出
            if text_width <= max_width:
                processed_subs.append({
                    'startTime': start_str,
                    'endTime': end_str,
                    'optimizedText': text
                })
                continue  # 处理下一个队列项

            # 否则需要拆分
            t_start = _parse_subtitle_time(start_str)
            t_end   = _parse_subtitle_time(end_str)
            duration = t_end - t_start
            if duration <= 0:
                # 畸形时间区间，跳过
                continue

            # --- 1. 找标点做最佳拆分点 ---
            best_split = -1
            min_offset = float('inf')
            for ch in split_chars:
                idx = 0
                while True:
                    pos = text.find(ch, idx)
                    if pos == -1:
                        break
                    # 考虑标点后面作为拆点
                    offset = abs(pos + 1 - len(text) / 2)
                    if offset < min_offset:
                        min_offset = offset
                        best_split = pos + 1
                    idx = pos + 1

            # --- 2. 如果没有标点就硬拆中间 ---
            if best_split == -1:
                best_split = len(text) // 2

            # 切成两段，去除首尾空白
            part1 = text[:best_split].strip()
            part2 = text[best_split:].strip()

            # 特殊情况：如果某段为空，就强制中点拆分一次
            if not part1 or not part2:
                mid = len(text) // 2
                part1 = text[:mid].strip()
                part2 = text[mid:].strip()

            # --- 3. 按字符比例分配时间 ---
            ratio1 = len(part1) / len(text)
            split_time_sec = t_start + duration * ratio1
            split_time_str = _format_time_for_ffmpeg(split_time_sec)

            # **改动点**：不再直接输出 part1，而是把两段都入队再检测
            segments_to_process.append((start_str, split_time_str, part1))
            segments_to_process.append((split_time_str, end_str,   part2))

        # end while

    # end for
    return processed_subs


def cover_video_area_simple(
    video_path: str,
    output_path: str,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: str = "black@1.0"
):
    """
    用 drawbox 滤镜在指定区域做纯色遮挡——极简、无坑版。
    color: 'RRGGBB@alpha' 格式，alpha 范围 0.0~1.0。
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    w, h = x2 - x1, y2 - y1

    vf = f"drawbox=x={x1}:y={y1}:w={w}:h={h}:color={color}:t=fill"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path
    ]
    print(f"[INFO] Running: {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed (code {proc.returncode}):\n{proc.stderr}")
    print(f"[SUCCESS] Output saved to {output_path}")

def cover_video_area_blur(
    video_path: str,
    output_path: str,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    blur_strength: int = 20
):
    """
    在指定区域应用模糊遮挡 - 最终修正版。
    修复了上一版本中因笔误导致的 "Unknown pixel format" 错误。
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    w, h = x2 - x1, y2 - y1

    temp_patch_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_f:
            temp_patch_file = temp_f.name

        print(f"[INFO] Pass 1: Creating blurred patch at {temp_patch_file}")

        # --- 第一阶段: 创建模糊补丁视频 (正确且不变) ---
        vf_pass1 = f"crop={w}:{h}:{x1}:{y1},boxblur={blur_strength}"
        cmd_pass1 = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
            "-vf", vf_pass1, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
            temp_patch_file
        ]

        print(f"[INFO] Running Pass 1: {' '.join(shlex.quote(c) for c in cmd_pass1)}")
        proc_pass1 = subprocess.run(cmd_pass1, capture_output=True, text=True, check=False)
        if proc_pass1.returncode != 0:
            raise RuntimeError(f"FFmpeg Pass 1 failed (code {proc_pass1.returncode}):\n{proc_pass1.stderr}")

        print(f"[INFO] Pass 2: Overlaying patch onto original video.")

        # --- 第二阶段: 叠加补丁 (修正笔误) ---
        # 1. overlay 滤镜中的 :format=yuv420 保持不变，这是绕过核心问题的关键
        # 2. 输出参数中的 -pix_fmt 改回正确的 yuv420p
        vf_pass2 = f"[0:v][1:v]overlay={x1}:{y1}:format=yuv420"
        cmd_pass2 = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-i", temp_patch_file,
            "-filter_complex", vf_pass2,
            "-c:a", "copy",
            "-c:v", "libx264",
            # [核心修正] 将错误的 "yuv4s20p" 改回正确的 "yuv420p"
            "-pix_fmt", "yuv420p",
            output_path
        ]

        print(f"[INFO] Running Pass 2: {' '.join(shlex.quote(c) for c in cmd_pass2)}")
        proc_pass2 = subprocess.run(cmd_pass2, capture_output=True, text=True, check=False)
        if proc_pass2.returncode != 0:
            raise RuntimeError(f"FFmpeg Pass 2 failed (code {proc_pass2.returncode}):\n{proc_pass2.stderr}")

        print(f"[SUCCESS] Output saved to {output_path}")

    finally:
        if temp_patch_file and os.path.exists(temp_patch_file):
            print(f"[INFO] Cleaning up temporary file: {temp_patch_file}")
            os.remove(temp_patch_file)

def add_subtitles_to_video(
        video_path: str,
        subtitles_info: list,
        output_path: str,
        font_path: str,
        font_size: int = 48,
        font_color: str = 'white',
        box_color: str = 'black@0.5',
        bottom_margin: int = 50,
        fixed_rect: list = [[180, 641], [1099, 699]]  # [[x1, y1], [x2, y2]]
) -> None:
    """
    将字幕信息“烧录”到视频中，并自动分割过长的字幕行，
    并在每条字幕出现的时段内，先绘制一个固定大小的矩形背景。

    :param video_path: 输入视频的路径。
    :param subtitles_info: 原始字幕信息列表。
    :param output_path: 输出视频的路径。
    :param font_path: 字体文件路径。
    :param font_size: 字体大小，默认 48。
    :param font_color: 字体颜色，默认白色。
    :param box_color: 半透明背景色，默认黑@0.5。
    :param bottom_margin: 距离底部的像素偏移，默认 50。
    :param fixed_rect: 固定矩形区域 [[x1,y1],[x2,y2]]，示例 [[180,641],[1099,699]]。
    """
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件未找到: {font_path}")

    # ------------------- [ 核心修改开始 ] -------------------

    # 1. 获取视频宽度以计算最大字幕宽度
    try:
        video_width, _ = get_video_dimensions(video_path)
        max_subtitle_width = video_width * 0.9
        print(f"视频宽度: {video_width}px, 字幕最大允许宽度: {max_subtitle_width:.0f}px")
    except (ValueError, FileNotFoundError) as e:
        print(f"警告: 无法获取视频尺寸，将不执行字幕分割。错误: {e}")
        processed_subtitles = subtitles_info
    else:
        # 2. 加载字体用于计算文本宽度
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            raise FileNotFoundError(f"无法加载字体文件，请检查路径和文件格式: {font_path}")

        # 3. 预处理字幕，分割过长行
        print("正在预处理字幕，检查并分割过长行...")
        processed_subtitles = _process_and_split_subtitles(
            subtitles_info,
            font,
            max_subtitle_width
        )
        print(f"字幕预处理完成。原始字幕数: {len(subtitles_info)}, 处理后字幕数: {len(processed_subtitles)}")

    # ------------------- [ 核心修改结束 ] -------------------

    # 为 ffmpeg 的滤镜语法格式化字体路径
    formatted_font_path = font_path.replace('\\', '/')
    if os.name == 'nt':
        formatted_font_path = formatted_font_path.replace(':', '\\:')

    # 计算固定矩形的位置和尺寸
    x1, y1 = fixed_rect[0]
    x2, y2 = fixed_rect[1]
    rect_w = x2 - x1
    rect_h = y2 - y1

    filters = []
    for sub in processed_subtitles:
        # 解析时间和文本
        start_time = _parse_subtitle_time(sub['startTime'])
        end_time   = _parse_subtitle_time(sub['endTime'])
        text       = _escape_ffmpeg_text(sub['optimizedText'])

        # 1) 先画固定大小的矩形
        drawbox = (
            f"drawbox="
            f"x={x1}:y={y1}:w={rect_w}:h={rect_h}:"
            f"color={box_color}:t=fill:"
            f"enable='between(t,{start_time},{end_time})'"
        )
        filters.append(drawbox)

        # 2) 再画字幕文字（关闭内置的 box）
        drawtext = (
            f"drawtext="
            f"fontfile='{formatted_font_path}':"
            f"text='{text}':"
            f"fontsize={font_size}:"
            f"fontcolor={font_color}:"
            f"x=(w-text_w)/2:"
            f"y=h-text_h-{bottom_margin}:"
            f"box=0:"
            f"enable='between(t,{start_time},{end_time})'"
        )
        filters.append(drawtext)

    if not filters:
        print("没有可烧录的字幕，将直接复制视频。")
        import shutil
        shutil.copy(video_path, output_path)
        return

    vf_arg = ",".join(filters)

    if not vf_arg:
        print("没有可烧录的字幕，将直接复制视频。")
        import shutil
        shutil.copy(video_path, output_path)
        return

    # 使用临时文件来保存滤镜链，避免命令行过长
    # tempfile.NamedTemporaryFile 在 with 块结束时会自动删除
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt", encoding='utf-8') as temp_filter_file:
        temp_filter_file.write(vf_arg)
        filter_script_path = temp_filter_file.name

    # 注意：在Windows上，临时文件路径可能包含反斜杠，需要处理
    formatted_filter_path = filter_script_path.replace('\\', '/')

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        # 使用 -filter_complex_script 或 -vf_script
        # -filter_complex_script 更通用，推荐使用
        "-filter_complex_script", formatted_filter_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path
    ]

    try:
        print("正在为视频添加字幕和固定矩形...")
        # 注意：这里 cmd 列表里不再包含那个超长的 vf_arg
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print(f"成功！已将带字幕的视频保存至: {output_path}")
    except FileNotFoundError:
        print("[错误] ffmpeg 未安装或未在系统 PATH 中。请先安装 ffmpeg。")
        raise
    except subprocess.CalledProcessError as e:
        print(f"[错误] ffmpeg 执行失败。返回码: {e.returncode}")
        print(f"FFMPEG 错误输出:\n{e.stderr}")
        # 可以在这里打印命令，帮助调试，但不要打印 vf_arg
        # print(f"执行的命令（不含滤镜内容）: {' '.join(cmd)}")
        print(f"滤镜脚本内容保存在: {filter_script_path}") # 告知用户可以检查这个文件
        raise
    finally:
        # 确保临时文件在任何情况下都被删除
        if os.path.exists(filter_script_path):
            os.remove(filter_script_path)

if __name__ == '__main__':
    # --- 使用示例 ---

    # 1. 准备输入和输出路径
    input_video = "output_video.mp4"
    output_with_subtitles = "output_with_subtitles.mp4"

    # 2. 准备字幕数据
    subtitle_data = [
    {
        "id": 1,
        "startTime": "00:00:00.376",
        "endTime": "00:00:03.812",
        "text": "AG让一追三击败KSG，实现跨赛季大场22连胜。",
        "optimizedText": "AG以三比一逆转KSG，达成跨赛季大场22连胜。",
        "old_startTime": "00:00:00.752",
        "old_endTime": "00:03.482",
        "forward_shift_ms": 376,
        "backward_shift_ms": 330,
        "duration": 3.436
    },
    {
        "id": 2,
        "startTime": "00:00:03.812",
        "endTime": "00:00:06.562",
        "text": "一诺成功将击杀记录刷新到三千三。",
        "optimizedText": "一诺选手成功将击杀数刷新到3300。",
        "old_startTime": "00:04.142",
        "old_endTime": "00:06.182",
        "forward_shift_ms": 330,
        "backward_shift_ms": 380,
        "duration": 2.75
    },
    {
        "id": 3,
        "startTime": "00:00:06.562",
        "endTime": "00:00:09.807",
        "text": "战文欲直接开启文艺复兴，拿小乔还有孙膑也能取胜。",
        "optimizedText": "张角想直接上演文艺复兴，用小乔孙膑阵容也能取胜。",
        "old_startTime": "00:06.942",
        "old_endTime": "00:09.522",
        "forward_shift_ms": 380,
        "backward_shift_ms": 285,
        "duration": 3.245
    },
    {
        "id": 4,
        "startTime": "00:00:09.807",
        "endTime": "00:00:12.042",
        "text": "手握满级号真的就可以为所欲为。",
        "optimizedText": "手持顶级号真的就能如此为所欲为。",
        "old_startTime": "00:10.092",
        "old_endTime": "00:11.752",
        "forward_shift_ms": 285,
        "backward_shift_ms": 290,
        "duration": 2.235
    },
    {
        "id": 5,
        "startTime": "00:00:12.042",
        "endTime": "00:00:14.937",
        "text": "都说被AG标记成强队的队伍下场都很惨。",
        "optimizedText": "据说被AG认证为强队的队伍结局都很惨。",
        "old_startTime": "00:12.332",
        "old_endTime": "00:14.652",
        "forward_shift_ms": 290,
        "backward_shift_ms": 285,
        "duration": 2.895
    },
    {
        "id": 6,
        "startTime": "00:00:14.937",
        "endTime": "00:00:17.217",
        "text": "在第一局输给KSG之后AG彻底觉醒。",
        "optimizedText": "在首局败给KSG之后AG便彻底觉醒。",
        "old_startTime": "00:15.222",
        "old_endTime": "00:17.062",
        "forward_shift_ms": 285,
        "backward_shift_ms": 155,
        "duration": 2.28
    },
    {
        "id": 7,
        "startTime": "00:00:17.217",
        "endTime": "00:00:19.732",
        "text": "第2局的长生直接把小乔玩成轰炸机。",
        "optimizedText": "第二局里长生直接把小乔玩成轰炸机。",
        "old_startTime": "00:17.372",
        "old_endTime": "00:19.462",
        "forward_shift_ms": 155,
        "backward_shift_ms": 270,
        "duration": 2.515
    },
    {
        "id": 8,
        "startTime": "00:00:19.732",
        "endTime": "00:00:21.842",
        "text": "配合大帅双大团战毁天灭地。",
        "optimizedText": "再配合大帅双大招团战毁天灭地。",
        "old_startTime": "00:20.002",
        "old_endTime": "00:21.662",
        "forward_shift_ms": 270,
        "backward_shift_ms": 180,
        "duration": 2.11
    },
    {
        "id": 9,
        "startTime": "00:00:21.842",
        "endTime": "00:00:24.277",
        "text": "这把可以说把长生的手法体现的淋漓尽致。",
        "optimizedText": "这局比赛将长生的操作展现得淋漓尽致。",
        "old_startTime": "00:22.022",
        "old_endTime": "00:24.082",
        "forward_shift_ms": 180,
        "backward_shift_ms": 195,
        "duration": 2.435
    },
    {
        "id": 10,
        "startTime": "00:00:24.277",
        "endTime": "00:00:26.842",
        "text": "说是目前KPL断档级中单丝毫不为过。",
        "optimizedText": "称他是目前KPL独一档中单也毫不为过。",
        "old_startTime": "00:24.472",
        "old_endTime": "00:26.552",
        "forward_shift_ms": 195,
        "backward_shift_ms": 290,
        "duration": 2.565
    },
    {
        "id": 11,
        "startTime": "00:00:26.842",
        "endTime": "00:00:29.682",
        "text": "另外一诺通过这局解锁了3300杀新里程碑。",
        "optimizedText": "此外一诺凭这局解锁了三千三百杀里程碑。",
        "old_startTime": "00:27.132",
        "old_endTime": "00:29.472",
        "forward_shift_ms": 290,
        "backward_shift_ms": 210,
        "duration": 2.84
    },
    {
        "id": 12,
        "startTime": "00:00:29.682",
        "endTime": "00:00:31.427",
        "text": "这个记录或许很难被打破。",
        "optimizedText": "这个纪录恐怕很难被打破。",
        "old_startTime": "00:29.892",
        "old_endTime": "00:31.252",
        "forward_shift_ms": 210,
        "backward_shift_ms": 175,
        "duration": 1.745
    },
    {
        "id": 13,
        "startTime": "00:00:31.427",
        "endTime": "00:00:33.437",
        "text": "到了第三局AG又打出了手法局。",
        "optimizedText": "来到第三局AG又打出了操作局。",
        "old_startTime": "00:31.602",
        "old_endTime": "00:32.962",
        "forward_shift_ms": 175,
        "backward_shift_ms": 475,
        "duration": 2.01
    },
    {
        "id": 14,
        "startTime": "00:00:33.437",
        "endTime": "00:00:36.732",
        "text": "双边阵容选出来也能打赢，或许只有目前的AG能做到。",
        "optimizedText": "选出双战边阵容照样能赢，可能只有现在的AG能办到。",
        "old_startTime": "00:33.912",
        "old_endTime": "00:36.422",
        "forward_shift_ms": 475,
        "backward_shift_ms": 310,
        "duration": 3.295
    },
    {
        "id": 15,
        "startTime": "00:00:36.732",
        "endTime": "00:00:39.287",
        "text": "大帅最后一大波大闪四个属实太C了。",
        "optimizedText": "大帅最后一波闪现大四个着实是太秀了。",
        "old_startTime": "00:37.042",
        "old_endTime": "00:39.112",
        "forward_shift_ms": 310,
        "backward_shift_ms": 175,
        "duration": 2.555
    },
    {
        "id": 16,
        "startTime": "00:00:39.287",
        "endTime": "00:00:41.092",
        "text": "不愧是AG本月唯一国服的含金量。",
        "optimizedText": "不愧是AG本月唯一国标的含金量。",
        "old_startTime": "00:39.462",
        "old_endTime": "00:40.942",
        "forward_shift_ms": 175,
        "backward_shift_ms": 150,
        "duration": 1.805
    },
    {
        "id": 17,
        "startTime": "00:00:41.092",
        "endTime": "00:00:42.927",
        "text": "各种刁钻的开团把KSG彻底打崩。",
        "optimizedText": "各种刁钻的开团让KSG彻底崩溃。",
        "old_startTime": "00:41.242",
        "old_endTime": "00:42.662",
        "forward_shift_ms": 150,
        "backward_shift_ms": 265,
        "duration": 1.835
    },
    {
        "id": 18,
        "startTime": "00:00:42.927",
        "endTime": "00:00:47.402",
        "text": "另外一诺的老夫子也很秀，能玩射手也能玩战边，难怪诺派会发扬光大。",
        "optimizedText": "此外一诺的老夫子也很秀，可当射手可当战边，难怪诺派能发扬光大。",
        "old_startTime": "00:43.192",
        "old_endTime": "00:46.902",
        "forward_shift_ms": 265,
        "backward_shift_ms": 500,
        "duration": 4.475
    },
    {
        "id": 19,
        "startTime": "00:01:32.844",
        "endTime": "00:01:37.114",
        "text": "随后AG又拿出复古阵容，孙膑加艾琳的下路组合许久没有见到。",
        "optimizedText": "接着AG又拿出复古阵容，孙膑配艾琳的下路组合很久没有见了。",
        "old_startTime": "01:33.344",
        "old_endTime": "01:36.914",
        "forward_shift_ms": 500,
        "backward_shift_ms": 200,
        "duration": 4.27
    },
    {
        "id": 20,
        "startTime": "00:01:37.114",
        "endTime": "00:01:39.614",
        "text": "但AG还是通过团战在前期取得优势。",
        "optimizedText": "但AG依旧通过团战在前期建立优势。",
        "old_startTime": "01:37.314",
        "old_endTime": "01:39.304",
        "forward_shift_ms": 200,
        "backward_shift_ms": 310,
        "duration": 2.5
    },
    {
        "id": 21,
        "startTime": "00:01:39.614",
        "endTime": "00:01:41.789",
        "text": "把孙膑体系的机动性发挥到了极致。",
        "optimizedText": "将孙膑体系的机动性发挥到了极限。",
        "old_startTime": "01:39.924",
        "old_endTime": "01:41.514",
        "forward_shift_ms": 310,
        "backward_shift_ms": 275,
        "duration": 2.175
    },
    {
        "id": 22,
        "startTime": "00:01:41.789",
        "endTime": "00:01:43.734",
        "text": "不过KSG这边的小控制很多。",
        "optimizedText": "然而KSG这边的小控制技能很多。",
        "old_startTime": "01:42.064",
        "old_endTime": "01:43.514",
        "forward_shift_ms": 275,
        "backward_shift_ms": 220,
        "duration": 1.945
    },
    {
        "id": 23,
        "startTime": "00:01:43.734",
        "endTime": "00:01:46.864",
        "text": "决胜时刻子阳的太乙站了出来，关键控制阻止了被AG速推。",
        "optimizedText": "决胜时刻子阳的太乙挺身而出，关键控制阻止了AG的速推。",
        "old_startTime": "01:43.954",
        "old_endTime": "01:46.614",
        "forward_shift_ms": 220,
        "backward_shift_ms": 250,
        "duration": 3.13
    },
    {
        "id": 24,
        "startTime": "00:01:46.864",
        "endTime": "00:01:48.969",
        "text": "可是AG的拉扯做的实在太好了。",
        "optimizedText": "但是AG的拉扯战术用得实在太好。",
        "old_startTime": "01:47.114",
        "old_endTime": "01:48.744",
        "forward_shift_ms": 250,
        "backward_shift_ms": 225,
        "duration": 2.105
    },
    {
        "id": 25,
        "startTime": "00:01:48.969",
        "endTime": "00:01:50.789",
        "text": "长生的火舞切后排非常果断。",
        "optimizedText": "长生的不知火舞切后排十分果断。",
        "old_startTime": "01:49.194",
        "old_endTime": "01:50.554",
        "forward_shift_ms": 225,
        "backward_shift_ms": 235,
        "duration": 1.82
    },
    {
        "id": 26,
        "startTime": "00:01:50.789",
        "endTime": "00:01:53.529",
        "text": "最后一波妖刀就算有三条命也没能力挽狂澜。",
        "optimizedText": "最后一波妖刀即使有三条命也无力回天。",
        "old_startTime": "01:51.024",
        "old_endTime": "01:53.114",
        "forward_shift_ms": 235,
        "backward_shift_ms": 415,
        "duration": 2.74
    },
    {
        "id": 27,
        "startTime": "00:01:53.529",
        "endTime": "00:01:56.009",
        "text": "萝卜这局的发挥确实被轩染对位。",
        "optimizedText": "萝卜这局的发挥确实被轩染对位压制。",
        "old_startTime": "01:53.944",
        "old_endTime": "01:55.774",
        "forward_shift_ms": 415,
        "backward_shift_ms": 235,
        "duration": 2.48
    },
    {
        "id": 28,
        "startTime": "00:01:56.009",
        "endTime": "00:01:57.844",
        "text": "直接成为了KSG这边的突破口。",
        "optimizedText": "他直接成为KSG战队这边的突破口。",
        "old_startTime": "01:56.244",
        "old_endTime": "01:57.484",
        "forward_shift_ms": 235,
        "backward_shift_ms": 360,
        "duration": 1.835
    },
    {
        "id": 29,
        "startTime": "00:01:57.844",
        "endTime": "00:02:00.014",
        "text": "虽然长生没有被评为这局MVP。",
        "optimizedText": "尽管长生本局没能被评选为MVP。",
        "old_startTime": "01:58.204",
        "old_endTime": "01:59.854",
        "forward_shift_ms": 360,
        "backward_shift_ms": 160,
        "duration": 2.17
    },
    {
        "id": 30,
        "startTime": "00:02:00.014",
        "endTime": "00:02:04.884",
        "text": "但是纵观整场比赛来看，长生的手法和意识都在大气层，而且在局内从来不刷KDA。",
        "optimizedText": "但纵观整场比赛的表现来看，长生的操作和意识属顶尖水准，且在游戏里从来不刷KDA。",
        "old_startTime": "02:00.174",
        "old_endTime": "02:04.384",
        "forward_shift_ms": 160,
        "backward_shift_ms": 500,
        "duration": 4.87
    },
    {
        "id": 31,
        "startTime": "00:02:37.793",
        "endTime": "00:02:41.898",
        "text": "在赢下这场比赛之后，AG的大场连胜记录已经到达22场。",
        "optimizedText": "在赢下这场对局之后，AG的大场连胜纪录已来到了22场。",
        "old_startTime": "02:38.293",
        "old_endTime": "02:41.523",
        "forward_shift_ms": 500,
        "backward_shift_ms": 375,
        "duration": 4.105
    },
    {
        "id": 32,
        "startTime": "00:02:41.898",
        "endTime": "00:02:43.178",
        "text": "更为恐怖的是。",
        "optimizedText": "更加令人恐惧的是。",
        "old_startTime": "02:42.273",
        "old_endTime": "02:42.923",
        "forward_shift_ms": 375,
        "backward_shift_ms": 255,
        "duration": 1.28
    },
    {
        "id": 33,
        "startTime": "00:02:43.178",
        "endTime": "00:02:45.883",
        "text": "曾经认为有望终结AG连胜的队伍，都在翻车或者被AG拿下。",
        "optimizedText": "那些曾被认为有望终结AG连胜的队伍，都翻车或者被AG拿下。",
        "old_startTime": "02:43.433",
        "old_endTime": "02:45.543",
        "forward_shift_ms": 255,
        "backward_shift_ms": 340,
        "duration": 2.705
    },
    {
        "id": 34,
        "startTime": "00:02:45.883",
        "endTime": "00:02:48.728",
        "text": "AG不仅有一诺这样越打越妖的老将。",
        "optimizedText": "AG不仅有一诺这样越战越勇的老将。",
        "old_startTime": "02:46.223",
        "old_endTime": "02:48.333",
        "forward_shift_ms": 340,
        "backward_shift_ms": 395,
        "duration": 2.845
    },
    {
        "id": 35,
        "startTime": "00:02:48.728",
        "endTime": "00:02:52.553",
        "text": "还有大帅轩染长生钟意这样强力的年轻选手，队伍五个位置完全没有破绽。",
        "optimizedText": "还有大帅轩染长生钟意等强力的年轻选手，队伍五个位置几乎毫无破绽。",
        "old_startTime": "02:49.123",
        "old_endTime": "02:52.223",
        "forward_shift_ms": 395,
        "backward_shift_ms": 330,
        "duration": 3.825
    },
    {
        "id": 36,
        "startTime": "00:02:52.553",
        "endTime": "00:02:54.913",
        "text": "甚至AG有全胜结束第2轮的可能。",
        "optimizedText": "甚至AG都有全胜结束第二轮的可能。",
        "old_startTime": "02:52.883",
        "old_endTime": "02:54.493",
        "forward_shift_ms": 330,
        "backward_shift_ms": 420,
        "duration": 2.36
    },
    {
        "id": 37,
        "startTime": "00:02:54.913",
        "endTime": "00:02:59.848",
        "text": "另外就在KPL夏季赛火爆进行的同时，隔壁CS的某牙钢盔杯也在火热进行中。",
        "optimizedText": "此外就在KPL夏季赛火热进行的同时，隔壁CS的虎牙钢盔杯也在火热进行中。",
        "old_startTime": "02:55.333",
        "old_endTime": "02:59.553",
        "forward_shift_ms": 420,
        "backward_shift_ms": 295,
        "duration": 4.935
    },
    {
        "id": 38,
        "startTime": "00:02:59.848",
        "endTime": "00:03:08.028",
        "text": "RA轻松打进决赛，CS boy透露钢盔杯还有下一届，能让更多CNCS年轻人展示自己，对CS感兴趣的小伙伴也可以去某牙观赛。",
        "optimizedText": "RA轻松打进决赛，CSBOY透露钢盔杯会有下一届，能让更多CNCS年轻人展示自己，对CS感兴趣的朋友们也能去虎牙观赛。",
        "old_startTime": "03:00.143",
        "old_endTime": "03:07.673",
        "forward_shift_ms": 295,
        "backward_shift_ms": 355,
        "duration": 8.18
    },
    {
        "id": 39,
        "startTime": "00:03:08.028",
        "endTime": "00:03:11.193",
        "text": "那么最后大家认为，AG的连胜记录会持续到多少场呢？",
        "optimizedText": "那么最后各位觉得，AG的连胜纪录会持续到多少场呢？",
        "old_startTime": "03:08.383",
        "old_endTime": "03:11.193",
        "forward_shift_ms": 355,
        "backward_shift_ms": 0,
        "duration": 3.165
    }
]
    try:
        # 尝试查找一个常见的系统字体
        font_file_path = ""
        if os.name == 'nt':  # Windows
            font_file_path = 'C:/Windows/Fonts/simhei.ttf'
            if not os.path.exists(font_file_path):
                font_file_path = 'C:/Windows/Fonts/msyh.ttc'
        elif os.name == 'posix':  # macOS or Linux
            if os.path.exists('/System/Library/Fonts/PingFang.ttc'):
                font_file_path = '/System/Library/Fonts/PingFang.ttc'  # macOS
            else:
                # 简单的Linux字体查找
                common_linux_fonts = [
                    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
                ]
                for font in common_linux_fonts:
                    if os.path.exists(font):
                        font_file_path = font
                        break

        if not font_file_path or not os.path.exists(font_file_path):
            raise FileNotFoundError("未能自动找到合适的系统字体。")

        print(f"自动检测到字体: {font_file_path}")

        # 4. 调用函数
        add_subtitles_to_video(
            video_path=input_video,
            subtitles_info=subtitle_data,
            output_path=output_with_subtitles,
            font_path=font_file_path,
            font_size=52,
            bottom_margin=60
        )

    except (FileNotFoundError, ValueError) as err:
        print(f"[主程序错误] 操作失败: {err}")
        print("\n[提示] 请确保：")
        print("1. `test.mp4` 文件存在于脚本相同目录下。")
        print("2. 你的系统中安装了 ffmpeg 并已添加到环境变量(PATH)。")
        print("3. 如果自动字体检测失败，请在代码中手动指定一个有效的中文字体路径。")


def get_duration_seconds(start_str, end_str):
    """
    计算两个FFmpeg时间格式字符串之间的秒数差。
    例如: "00:01:15.250" 和 "00:00:10.100"
    """

    def time_to_seconds(t_str):
        """将FFmpeg时间格式字符串转换为总秒数"""
        try:
            # 尝试按 HH:MM:SS.ms 格式解析
            h, m, s_ms = t_str.split(':')
            s, ms = (s_ms.split('.') + ['0'])[:2]  # 处理没有毫秒的情况
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        except ValueError:
            # 如果上面的解析失败，可能是一个简单的秒数
            return float(t_str)

    start_seconds = time_to_seconds(start_str)
    end_seconds = time_to_seconds(end_str)

    return end_seconds - start_seconds


def re_edit_video_ffmpeg(video_path, time_segments, output_path="output_video_ffmpeg.mp4"):
    """
    使用 FFmpeg 根据给定的时间段列表重新剪辑视频。

    此版本已修正以下问题：
    1. 非时序拼接导致的时间戳混乱问题 (通过全部重新编码解决)。
    2. 剪辑不精确导致最终视频过长和内容重复的问题 (通过使用 -t 时长参数解决)。
    """
    if not time_segments:
        print("[ERROR] 时间段列表为空，操作中止。")
        return

    if not os.path.exists(video_path):
        print(f"[ERROR] 视频文件未找到: {video_path}")
        return

    # 使用临时目录来存放剪辑的片段，程序结束后会自动清理
    with tempfile.TemporaryDirectory() as temp_dir:
        clip_files = []

        print("[INFO] 开始剪辑视频片段...")
        for i, segment in enumerate(time_segments):
            start_time = segment['original_start_time']
            end_time = segment['original_end_time']

            # 核心修正 1: 计算片段的精确时长
            duration = get_duration_seconds(start_time, end_time)
            if duration <= 0:
                print(f"[WARNING] 片段 {i + 1} 的时长为零或负数，将跳过。开始: {start_time}, 结束: {end_time}")
                continue

            temp_clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")

            print(
                f"[INFO] 正在创建片段 {i + 1}/{len(time_segments)} (方法: 重新编码): 从 {start_time} 开始，时长 {duration:.3f} 秒")

            # 核心修正 2: 使用 -t (时长) 替代 -to (结束时间)，并对所有片段重新编码
            cmd_clip = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", start_time,  # 快速定位到开始时间
                "-i", video_path,
                "-t", str(duration),  # 精确指定剪辑时长
                "-c:v", "libx264",  # 重新编码视频以重置时间戳
                "-c:a", "aac",  # 重新编码音频
                "-preset", "fast",  # 使用较快的编码预设，平衡速度和质量
                "-crf", "22",  # 合理的质量参数 (18-28, 数字越小质量越高)
                temp_clip_path
            ]

            try:
                # 运行命令，并捕获输出用于调试
                result = subprocess.run(cmd_clip, check=True, capture_output=True, text=True, encoding='utf-8')
                clip_files.append(temp_clip_path)
            except subprocess.CalledProcessError as e:
                # 如果FFmpeg执行失败，打印详细的错误信息
                print(f"    [ERROR] 创建片段 {i + 1} 失败 (返回码 {e.returncode})。")
                print(f"    [DEBUG] 失败的命令: {' '.join(e.cmd)}")
                if e.stderr:
                    print(f"    [DEBUG] FFmpeg Stderr:\n---(start) ---\n{e.stderr.strip()}\n--- (end) ---")
                print(f"[WARNING] 将跳过此时间段继续。")
                continue

        # --- 后续拼接逻辑 ---
        if not clip_files:
            print("\n[ERROR] 未能成功创建任何剪辑片段，拼接操作中止。")
            return

        print(f"\n[INFO] 成功创建 {len(clip_files)} 个片段，现在开始拼接...")

        # 创建一个文本文件，列出所有要拼接的片段
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list_path, 'w', encoding='utf-8') as f:
            for clip_path in clip_files:
                # 使用绝对路径并处理反斜杠，确保在所有系统上都安全
                safe_clip_path = os.path.abspath(clip_path).replace('\\', '/')
                f.write(f"file '{safe_clip_path}'\n")

        # 使用 concat demuxer 进行快速拼接
        # 因为所有片段都已重新编码为相同格式，所以使用-c copy拼接会非常快且可靠
        cmd_concat = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path
        ]

        try:
            subprocess.run(cmd_concat, check=True, capture_output=True, text=True, encoding='utf-8')
            print(f"[SUCCESS] 视频已成功合并并保存至: {output_path}")
        except subprocess.CalledProcessError as e:
            # 如果流复制拼接失败（理论上不应该发生，但作为备用方案），尝试重新编码拼接
            print(f"[WARNING] 使用流复制进行拼接失败，将尝试重新编码拼接。")
            print(f"    [DEBUG] 失败的命令: {' '.join(e.cmd)}")
            if e.stderr:
                print(f"    [DEBUG] FFmpeg Stderr:\n---(start) ---\n{e.stderr.strip()}\n--- (end) ---")

            cmd_concat_recode = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "fast",
                "-crf", "22",
                output_path
            ]
            try:
                subprocess.run(cmd_concat_recode, check=True, capture_output=True, text=True, encoding='utf-8')
                print(f"[SUCCESS] 视频已成功合并（通过重新编码）并保存至: {output_path}")
            except subprocess.CalledProcessError as e3:
                print(f"[FATAL] 重新编码拼接也失败了。")
                if e3.stderr:
                    print(f"    [DEBUG] FFmpeg Stderr:\n---(start) ---\n{e3.stderr.strip()}\n--- (end) ---")


def get_video_duration_seconds(video_path: str) -> float | None:
    """
    使用 ffprobe 获取视频时长（秒）。

    Args:
        video_path: 视频文件的路径。

    Returns:
        一个浮点数表示的视频时长（秒），如果无法获取则返回 None。
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        return None

    # 构建 ffprobe 命令
    # -v error: 只在发生错误时打印日志
    # -show_entries format=duration: 只显示 'format' 部分的 'duration' 字段
    # -of default=noprint_wrappers=1:nokey=1: 使用默认输出格式，但不打印包装器和键，只输出值
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    print(f"[INFO] Running: {' '.join(shlex.quote(c) for c in cmd)}")

    # 执行命令
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # 检查 ffprobe 是否成功执行
    if proc.returncode != 0:
        print(f"[ERROR] ffprobe failed (code {proc.returncode}) for file '{video_path}':")
        print(proc.stderr)
        return None

    # 解析输出
    try:
        # proc.stdout 应该是类似 "123.456000\n" 的字符串
        duration_str = proc.stdout.strip()
        if not duration_str:
            print(f"[ERROR] ffprobe returned empty duration for file '{video_path}'.")
            return None
        return float(duration_str)
    except ValueError:
        print(f"[ERROR] Could not parse duration from ffprobe output: '{proc.stdout}'")
        return None

def _get_image_dimensions(image_path: str) -> tuple[int, int] or None:
    # (此辅助函数无需修改)
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', image_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        data = json.loads(result.stdout)
        return data['streams'][0]['width'], data['streams'][0]['height']
    except Exception as e:
        print(f"错误: 无法获取图片尺寸 '{image_path}'.")
        print(f"具体错误: {e}")
        return None


def create_enhanced_cover(
        input_image_path: str,
        output_image_path: str,
        text_lines: list[str],
        font_path='C:/Windows/Fonts/msyhbd.ttc',
        position: str = 'top_third',
        color_theme: str = 'auto',
        font_size_ratio: float = 1.0,
        line_spacing_ratio: float = 1.4,
        overwrite: bool = True
) -> str or None:
    if not all([os.path.exists(input_image_path), os.path.exists(font_path)]):
        print("错误: 输入文件或字体文件未找到。")
        return None

    dimensions = _get_image_dimensions(input_image_path)
    if not dimensions: return None
    img_w, img_h = dimensions
    true_high = int(img_w * 9 / 16)

    if not text_lines:
        print("警告: 未提供任何文字，将直接复制图片。")
        if overwrite or not os.path.exists(output_image_path):
            shutil.copy(input_image_path, output_image_path)
        return output_image_path

    # !! 关键修改 1: 优化颜色主题，并增强阴影对比度 !!
    color_themes = {
        # # 主题1: 经典白字黑边 (最通用，最清晰)
        'classic_white': {'fontcolor': 'White', 'shadowcolor': 'black@0.8'},
        # # 主题2: 活力黄黑配 (最醒目，适合娱乐内容)
        'vibrant_yellow': {'fontcolor': '#FFD700', 'shadowcolor': 'black@0.85'},
        'cyber_cyan': {'fontcolor': '0x00FFFF', 'shadowcolor': 'black@0.4'},
        'energetic_orange': {'fontcolor': '#FF6347', 'shadowcolor': 'white@0.8'},

    }

    # 如果指定的主题不存在，或为 'auto'，则从预设中随机选择
    if color_theme not in color_themes or color_theme == 'auto':
        # 默认随机选择，但可以优先选择最经典的
        # chosen_theme = color_themes['classic_white']
        chosen_theme = random.choice(list(color_themes.values()))
    else:
        chosen_theme = color_themes[color_theme]

    longest_line = max(text_lines, key=len)
    target_text_width = img_w * 0.95
    estimated_char_width_ratio = 1.0
    font_size = int(min((target_text_width / len(longest_line)), img_h / 4) * font_size_ratio)

    # !! 关键修改 2: 增加阴影偏移量，模拟更厚的描边效果 !!
    # 将偏移量从原来的5%提升到8%
    shadow_offset = max(2, int(font_size * 0.06))

    line_height = int(font_size * line_spacing_ratio)
    total_text_height = line_height * (len(text_lines) - 1) + font_size

    escaped_font_path = font_path.replace(':', '\\:') if os.name == 'nt' else font_path

    position_map = {'center': img_h / 2, 'top_third': (img_h / 2 - true_high / 2 + font_size /2), 'bottom_third': img_h * 0.75}
    block_y_center = position_map.get(position, img_h * 0.5)  # 默认居中
    start_y = block_y_center - total_text_height / 2

    filters = []
    for i, line in enumerate(text_lines):
        line_y = start_y + i * line_height
        x_expr = '(w-text_w)/2'

        drawtext_options = {
            'fontfile': f"'{escaped_font_path}'",
            'text': f"'{line.replace(':', '\\:').replace('%', '\\%').replace('\'', '')}'",
            'fontsize': str(font_size),
            'fontcolor': chosen_theme['fontcolor'],
            'x': x_expr,
            'y': str(line_y),
            'shadowcolor': chosen_theme['shadowcolor'],
            'shadowx': str(shadow_offset),
            'shadowy': str(shadow_offset)
        }
        filters.append("drawtext=" + ":".join(f"{k}={v}" for k, v in drawtext_options.items()))

    vf_string = ",".join(filters)
    command = ['ffmpeg', '-i', input_image_path, '-vf', vf_string]
    if overwrite: command.append('-y')
    command.append(output_image_path)

    print(f"主题: {chosen_theme}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"🎉 成功! 优化后的封面已保存到 '{output_image_path}'")
        return output_image_path
    except subprocess.CalledProcessError as e:
        print("FFMPEG 执行失败!")
        print(f"错误码: {e.returncode}")
        print("FFMPEG 输出 (stderr):")
        print(e.stderr)
        return None
