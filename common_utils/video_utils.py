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
import subprocess
import json

from common_utils.common_utils import time_to_ms


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
                if output_size >= input_size:
                    # 成功且大小正常，退出循环
                    return
                else:
                    print(
                        f"[警告] 第 {attempt} 次生成的视频大小 ({output_size}) 小于输入视频大小 ({input_size})，重试中...")
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


import subprocess
import json
import shlex

def _get_video_resolution(video_path: str):
    """
    调用 ffprobe 自动获取视频宽高（像素）。
    返回 (width, height)。
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 获取分辨率失败：{proc.stderr.strip()}")
    info = json.loads(proc.stdout)
    stream = info.get("streams", [{}])[0]
    return int(stream["width"]), int(stream["height"])


def cover_video_area_gently(
        video_path: str,
        output_path: str,
        top_left: tuple[int,int],
        bottom_right: tuple[int,int],
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
        # luma 模糊 + 动态 chroma 上限
        max_chroma = height // 4
        chroma = min(strength, max_chroma)
        effect = f"boxblur=luma_radius={strength}:lr={strength}" \
                 f":chroma_radius={chroma}:cr={chroma}"
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
        proc = subprocess.run(cmd, capture_output=True, text=True)
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


def add_subtitles_to_video(
        video_path: str,
        subtitles_info: list,
        output_path: str,
        font_path: str,
        font_size: int = 48,
        font_color: str = 'white',
        box_color: str = 'black@0.5',
        bottom_margin: int = 50
) -> None:
    """
    将字幕信息“烧录”到视频中。

    :param video_path: 输入视频的路径。
    :param subtitles_info: 字幕信息列表，每个元素是一个包含 'startTime', 'endTime', 'text' 的字典。
    :param output_path: 输出视频的路径。
    :param font_path: 字体文件的路径（.ttf, .otf, .ttc 等）。对于中文字幕，必须提供支持中文的字体。
    :param font_size: 字幕的字号。
    :param font_color: 字幕颜色（例如 'white', 'black', '#FF0000'）。
    :param box_color: 字幕背景框的颜色和透明度（例如 'black@0.5' 表示半透明黑色）。
    :param bottom_margin: 字幕距离视频底部的像素边距。
    :raises ValueError: 如果字幕信息或字体路径无效。
    :raises subprocess.CalledProcessError: 如果 ffmpeg 命令执行失败。
    :raises FileNotFoundError: 如果 ffmpeg 未安装或字体文件不存在。
    """
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件未找到: {font_path}")

    # ------------------- [ 核心修改点 ] -------------------
    # 为ffmpeg的滤镜语法格式化字体路径
    # 1. 将所有反斜杠 \ 替换为正斜杠 / (更通用的路径分隔符)
    formatted_font_path = font_path.replace('\\', '/')
    # 2. 对Windows路径中的盘符冒号 : 进行转义，防止ffmpeg滤镜解析错误
    if os.name == 'nt':
        # 在Python中，'\\:' 会在字符串中生成 '\:'，这正是ffmpeg需要的
        formatted_font_path = formatted_font_path.replace(':', '\\:')
    # ----------------------------------------------------

    drawtext_filters = []
    for sub in subtitles_info:
        try:
            start_time = _parse_subtitle_time(sub['startTime'])
            end_time = _parse_subtitle_time(sub['endTime'])
            text = _escape_ffmpeg_text(sub['optimizedText'])
        except (KeyError, ValueError) as e:
            raise ValueError(f"字幕条目格式错误: {sub}. 错误: {e}")

        # 构建单个字幕的 drawtext 滤镜
        filter_part = (
            f"drawtext="
            f"fontfile='{formatted_font_path}':"
            f"text='{text}':"
            f"fontsize={font_size}:"
            f"fontcolor={font_color}:"
            f"x=(w-text_w)/2:"  # 水平居中
            f"y=h-text_h-{bottom_margin}:"  # 修正 y 坐标以使用 text_h (th 是 text_h 的别名)
            f"box=1:boxcolor={box_color}:boxborderw=15:"  # 启用背景框，并设置边距
            f"enable='between(t,{start_time},{end_time})'"
        )
        drawtext_filters.append(filter_part)

    # 将所有 drawtext 滤镜用逗号连接成一个大的视频滤镜（-vf）参数
    final_filter_string = ",".join(drawtext_filters)

    cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出文件
        "-loglevel", "error",  # 只在出错时显示日志
        "-i", video_path,
        "-vf", final_filter_string,
        "-c:v", "libx264",  # 需要重新编码视频以烧录字幕
        "-pix_fmt", "yuv420p",  # 保证兼容性
        "-c:a", "copy",  # 复制音频流，不重新编码，速度更快
        output_path
    ]

    try:
        # 打印一个更简洁的命令，避免终端过长
        print(f"正在为视频添加字幕...")
        # print(f"执行命令: {' '.join(cmd)}") # 如果需要调试，可以取消这行注释

        # 使用 capture_output=True 来捕获 stderr，便于调试
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print(f"成功！已将带字幕的视频保存至: {output_path}")
    except FileNotFoundError:
        print("[错误] ffmpeg 未安装或未在系统 PATH 中。请先安装 ffmpeg。")
        raise
    except subprocess.CalledProcessError as e:
        print(f"[错误] ffmpeg 执行失败。返回码: {e.returncode}")
        print(f"FFMPEG 错误输出:\n{e.stderr}")
        raise

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