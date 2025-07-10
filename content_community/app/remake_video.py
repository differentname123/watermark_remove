# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/7/10 1:46
:last_date:
    2025/7/10 1:46
:description:
    重制视频
"""
import os
import time

from LLM.gemini import get_llm_content_gemini_flash_video
from common_utils.common_utils import string_to_object, optimize_subtitle_timing
from common_utils.split_scenes import find_and_split_scenes
from common_utils.tts.paddle_demo import synthesize_and_get_duration
from common_utils.video_utils import cover_video_area_gently, add_subtitles_to_video
from paddlespeech.cli.tts.infer import TTSExecutor

import json

from common_utils.video_utils1 import redub_video_with_ffmpeg


def get_owner_speech(video_path):
    """
    获取视频中的主人公语音片段。以及相应的语音
    """
    prompt = """
    你是一名专业的音频处理AI，任务是进行说话人识别和语音转写。

    # 任务背景
    - 我是视频创作者，我将为你提供一份带有时间戳的语音转写初稿或音频文件。
    - 内容中混合了我的旁白、其他人的声音、以及外部视频片段的声音。

    # 任务目标
    1.  **识别主体**：在所有声音中，只识别并提取出属于“我”（视频创作者）的旁白部分。
    2.  **内容筛选**：完全忽略所有其他人声、背景音、以及非我本人说出的语句。
    3.  **精准对齐**：将我说的每一句旁白，都切分成一个符合自然语义的完整短句。每一句都必须带有精确到毫秒的起始和结束时间戳。
    4.  **验证校准**：如果给出了时间区间，请验证该区间是否准确对应我的声音，并进行必要的校准。

    # 输出要求
    - **格式**：最终结果必须是一个纯净、合法的 JSON 数组 (`Array of Objects`)。
    - **内容**：你的回答**必须且只能是**这个 JSON 数组本身，绝对不能包含任何解释性文字、注释、Markdown 标记（例如 ```json）或任何非 JSON 内容。
    - **结构**：数组中的每个对象代表我的一句旁白，包含以下四个字段：
        - `id`: (Number) 序号，从 1 开始递增。
        - `startTime`: (String) 开始时间，格式为 `HH:MM:SS.mmm`。
        - `endTime`: (String) 结束时间，格式为 `HH:MM:SS.mmm`。
        - `text`: (String) 旁白文本内容。

    # JSON 格式示例
    [
      {
        "id": 1,
        "startTime": "00:00:03.125",
        "endTime": "00:00:05.890",
        "text": "欢迎来到我的视频。"
      },
      {
        "id": 2,
        "startTime": "00:00:07.500",
        "endTime": "00:00:10.000",
        "text": "今天我们来聊一个重要话题。"
      }
    ]
    """

    # raw = get_llm_content_gemini_flash_video(prompt=prompt, video_path=video_path)
    # result = string_to_object(raw)
    # # 将result保存到result.json
    # with open('result.json', 'w', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)

    # 直接读取result.json文件
    with open('result.json', 'r', encoding='utf-8') as f:
        result = json.load(f)

    optimized_subtitles = optimize_subtitle_timing(result)
    with open('result1.json', 'w', encoding='utf-8') as f:
        json.dump(optimized_subtitles, f, ensure_ascii=False, indent=4)
    return optimized_subtitles

def cover_subtitle(video_path, output_path,top_left, bottom_right):
    """
    覆盖视频中的字幕
    """
    start_time = time.time()

    vid_w, vid_h = cover_video_area_gently(
        video_path=video_path,
        output_path=output_path,
        top_left=top_left,
        bottom_right=bottom_right,
        mode='blur',
        strength=30  # 模糊强度，可以调整
    )
    print(f"覆盖字幕区域完成，输出文件: {output_path} 耗时: {time.time() - start_time:.2f} 秒")
    return vid_w, vid_h

def gen_new_audio(optimized_subtitles):
    output_dir = 'output_audio'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tts_engine = TTSExecutor()
    for subtitle in optimized_subtitles:
        output_file = os.path.join(output_dir, f"{subtitle['id']}.wav")
        text_to_speak = subtitle['optimizedText']
        audio_length = synthesize_and_get_duration(
            tts_executor=tts_engine,
            text=text_to_speak,
            output_path=output_file
        )
        subtitle['outputPath'] = output_file
        subtitle['trimmedDuration'] = audio_length
        print(f"生成音频 {subtitle['id']} 完成，{text_to_speak} 时长: {audio_length:.2f} 秒")
    # 保存优化后的字幕
    with open('optimized_subtitles.json', 'w', encoding='utf-8') as f:
        json.dump(optimized_subtitles, f, ensure_ascii=False, indent=4)

    # 读取optimized_subtitles
    with open('optimized_subtitles.json', 'r', encoding='utf-8') as f:
        optimized_subtitles = json.load(f)
    return optimized_subtitles

def add_subtitle(input_video, subtitle_data, output_with_subtitles, bottom_margin, font_size):
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
            font_size=font_size,
            bottom_margin=bottom_margin
        )

    except (FileNotFoundError, ValueError) as err:
        print(f"[主程序错误] 操作失败: {err}")
        print("\n[提示] 请确保：")
        print("1. `test.mp4` 文件存在于脚本相同目录下。")
        print("2. 你的系统中安装了 ffmpeg 并已添加到环境变量(PATH)。")
        print("3. 如果自动字体检测失败，请在代码中手动指定一个有效的中文字体路径。")

def remake_video(video_path):
    """
    重制视频
    """
    # 获取主人公语音片段
    owner_speech_list = get_owner_speech(video_path)
    top_left = (35, 646)
    bottom_right = (1139, 720)
    # 覆盖字幕区域
    covered_video_path = video_path.replace('.mp4', '_covered.mp4')
    vid_w, vid_h = cover_subtitle(video_path, covered_video_path, top_left, bottom_right)

    add_subtitle_output_path = covered_video_path.replace('.mp4', '_with_subtitles.mp4')
    font_size = bottom_right[1] - top_left[1]
    font_size = int(font_size * 0.8)
    bottom_margin = vid_h - bottom_right[1] + int(int(bottom_right[1] - top_left[1]) * 0.1)
    add_subtitle(covered_video_path, owner_speech_list, add_subtitle_output_path, bottom_margin=bottom_margin, font_size=font_size)

    # 生成新的音频
    optimized_subtitles = gen_new_audio(owner_speech_list)

    # 使用ffmpeg重制视频
    redub_video_with_ffmpeg(add_subtitle_output_path, optimized_subtitles, output_path='output_with_new_audio.mp4')

if __name__ == '__main__':
    remake_video('test.mp4')
