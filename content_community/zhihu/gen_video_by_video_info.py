import copy
import json
import os
import pathlib
import random
import re
import shutil
import time

from PIL import Image, UnidentifiedImageError

from bs4 import BeautifulSoup, Tag, NavigableString

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import requests
from typing import List, Dict

from LLM.gemini import analyze_images_gemini
from common_utils.common_utils import save_json, download_public_image, read_json, string_to_object, ms_to_time
from common_utils.tts.edge_tts_utils import generate_audio_and_get_duration_sync
from common_utils.video_utils import add_subtitles_to_video, merge_videos_ffmpeg
from common_utils.video_utils1 import redub_video_with_ffmpeg
from common_utils.video_utils2 import add_bgm_to_video
from common_utils.video_utils_cut import create_video_from_image_auto_select
from content_community.zhihu.gen_zhihu_video_info import gen_video_final_info


def prepare_all_image(content_list, origin_image_path_dir, output_image_path_dir):
    """
    准备好所有需要的图片，每个条目不存在或报错时使用上一张。
    """
    image_info = {}
    # 创建输出目录
    os.makedirs(output_image_path_dir, exist_ok=True)

    last_image_path = None
    for i, content in enumerate(content_list):
        image_name = content.get("配图", "")
        origin_image_path = pathlib.Path(origin_image_path_dir) / image_name
        output_image_path = pathlib.Path(output_image_path_dir) / f"{i}.jpg"

        used_image_path = None
        # 尝试使用当前图片
        if origin_image_path.exists():
            try:
                img = Image.open(origin_image_path)
                img.save(output_image_path)
                used_image_path = output_image_path
                print(f"已复制图片 {origin_image_path} 到 {output_image_path}")
            except UnidentifiedImageError:
                print(f"错误：无法识别图片 {origin_image_path}，将使用上一张图片。")
        else:
            print(f"错误：原始图片文件 {origin_image_path} 不存在，将使用上一张图片。")

        # 如果当前图片不合格且有上一张，复用上一张
        if used_image_path is None:
            if last_image_path is not None and last_image_path.exists():
                try:
                    img = Image.open(last_image_path)
                    img.save(output_image_path)
                    used_image_path = output_image_path
                    print(f"已复用上一张图片 {last_image_path} 到 {output_image_path}")
                except Exception as e:
                    print(f"错误：复用上一张图片失败 ({e})。输出路径: {output_image_path}")
            else:
                print(f"警告：没有可用的上一张图片，条目 {i} 未设置图片。")

        # 记录使用的图片路径
        if used_image_path is not None:
            image_info[str(i)] = {"image_path": str(output_image_path.resolve())}
        else:
            image_info[str(i)] = {"image_path": None}

        # 更新上一张图片路径
        if used_image_path is not None:
            last_image_path = used_image_path

    return image_info



def gen_part_video(image_path, audio_path, output_video_path, duration, text_to_speak, short_text_to_speak):
    """
    生成单个视频片段
    """
    if not os.path.exists(image_path):
        print(f"错误：图片文件 {image_path} 不存在，无法生成视频片段。")
        return

    if not os.path.exists(audio_path):
        print(f"错误：音频文件 {audio_path} 不存在，无法生成视频片段。")
        return
    if not os.path.exists(output_video_path):
        output_video_path = pathlib.Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
    image_output_video_path = image_path.replace(".jpg", ".mp4")
    # 图片生成视频
    create_video_from_image_auto_select(image_path=image_path, output_path=image_output_video_path, duration=duration)


    # 为视频增加语音
    audio_output_video_path = image_output_video_path.replace(".mp4", "_audio.mp4")
    segments_info = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(duration * 1000),
        'outputPath': audio_path,
        'trimmedDuration': duration,
    }]
    redub_video_with_ffmpeg(image_output_video_path, segments_info, output_path=audio_output_video_path)

    # 为视频增加字幕
    subtitle_data = [{
        'startTime': "00:00:00.000",
        'endTime': ms_to_time(duration * 1000),
        'optimizedText': text_to_speak
    }]
    subtitle_data_video_path = audio_output_video_path.replace(".mp4", "_subtitles.mp4")
    add_subtitles_to_video(
        video_path=audio_output_video_path,
        subtitles_info=subtitle_data,
        output_path=subtitle_data_video_path,
        font_size=70,
        bottom_margin=30
    )

    if len(text_to_speak) > 30:
        # 增加简略文案
        subtitle_data = [{
            'startTime': ms_to_time(duration * 500),
            'endTime': ms_to_time(duration * 1000),
            'optimizedText': short_text_to_speak
        }]
        add_subtitles_to_video(
            video_path=subtitle_data_video_path,
            subtitles_info=subtitle_data,
            output_path=output_video_path,
            font_color='#FFD700',
            font_size=80,
            bottom_margin=1000
        )
    else:
        # 将subtitle_data_video_path复制到output_video_path
        shutil.copy(subtitle_data_video_path, output_video_path)
    print(f"生成视频片段：{output_video_path}")

def prepare_video(video_info, output_video_path_dir):
    """
    生成相应的视频片段
    """
    video_part_info = {}
    image_info = video_info.get("image_info", {})
    audio_info = video_info.get("audio_info", {})
    for k, value in audio_info.items():
        audio_path = value.get("audio_path", "")
        duration = value.get("duration", 0)
        text_to_speak = value.get("text_to_speak", "")
        short_text_to_speak = value.get("short_text_to_speak", "")
        if not os.path.exists(audio_path):
            print(f"错误：音频文件 {audio_path} 不存在，无法生成视频片段。")
            continue

        image_path = image_info.get(k, {}).get("image_path", "")
        if not os.path.exists(image_path):
            print(f"错误：图片文件 {image_path} 不存在，无法生成视频片段。")
            continue

        output_video_path = output_video_path_dir / f"{k}.mp4"
        gen_part_video(
            image_path=image_path,
            audio_path=audio_path,
            text_to_speak=text_to_speak,
            short_text_to_speak=short_text_to_speak,
            output_video_path=str(output_video_path),
            duration=duration
        )
        video_part_info[k] = {
            "id": k,
            "image_path": image_path,
            "audio_path": audio_path,
            "text_to_speak": text_to_speak,
            "short_text_to_speak": short_text_to_speak,
            "duration": duration,
            "video_path": str(output_video_path.resolve())
        }
    return video_part_info

def prepare_all_audio(content_list,voice_name, audio_path_dir):
    audio_info = {}
    for i, content in enumerate(content_list):
        audio_file_path = audio_path_dir / f"{i}.mp3"
        text_to_speak = content.get("文案", "")
        short_text_to_speak = content.get("简略文案", "")
        audio_length = generate_audio_and_get_duration_sync(
            text=text_to_speak,
            output_filename=str(audio_file_path),
            voice_name=voice_name,
            trim_silence=False,
            rate="+15%",
            pitch='+10Hz',
        )
        abs_audio_path = str(audio_file_path.resolve())
        audio_info[f'{i}'] = {
            "id": i,
            "audio_path": abs_audio_path,
            "text_to_speak": text_to_speak,
            "short_text_to_speak": short_text_to_speak,
            "duration": audio_length
        }
    return audio_info

def merge_all_videos(video_part_info, output_path):
    """
    合并所有视频片段
    """
    video_path_list = []
    for k, value in video_part_info.items():
        video_path = value.get("video_path", "")
        if os.path.exists(video_path):
            video_path_list.append(video_path)
        else:
            print(f"警告：视频片段 {video_path} 不存在，跳过。")

    if video_path_list:
        merge_videos_ffmpeg(video_path_list, output_path=output_path)
        print(f"合并视频完成，输出路径：{output_path}")
    else:
        print("没有可用的视频片段，无法合并。")

def gen_video_by_video_info(video_info_file, bgm_library_path=r"W:\project\python_project\watermark_remove\content_community\app\bgm_audio"):
    """
    通过视频文案脚本生成最终的完整视频
    """
    video_info = read_json(video_info_file)
    bgm_id = video_info.get("recommendations", {}).get("bgm", {}).get("id", "").replace(".mp4", ".wav")
    bgm_path = f"{bgm_library_path}/{bgm_id}"
    if not os.path.exists(bgm_path):
        print(f"错误：背景音乐文件 '{bgm_path}' 不存在。")
        return
    voice_name = video_info.get("recommendations", {}).get("voice", {}).get("voice_name", "zh-CN-YunyangNeural")

    base_dir = pathlib.Path(video_info_file).parent
    output_path_dir = base_dir / "video_output"
    # 确保目录存在
    output_path_dir.mkdir(parents=True, exist_ok=True)

    gen_video_info_path = output_path_dir / "gen_video_info.json"
    gen_video_info = read_json(str(gen_video_info_path))


    content_list = video_info.get("optimized_content", "")
    # 生成配音
    audio_path_dir = output_path_dir / "audio"
    audio_path_dir.mkdir(parents=True, exist_ok=True)
    if 'audio_info' not in gen_video_info:
        audio_info = prepare_all_audio(content_list, voice_name, audio_path_dir)
        gen_video_info['audio_info'] = audio_info
        save_json(str(gen_video_info_path), gen_video_info)


    print("配音生成完成，开始生成视频...")
    if 'image_info' not in gen_video_info:
        image_info = prepare_all_image(content_list, base_dir / "images", output_path_dir / "images")
        gen_video_info['image_info'] = image_info
        save_json(str(gen_video_info_path), gen_video_info)
    print("图片准备完成，开始生成视频片段...")

    # 准备生成视频片段
    if 'video_part_info' not in gen_video_info:
        video_part_info = prepare_video(gen_video_info, output_path_dir / "video_parts")
        gen_video_info['video_part_info'] = video_part_info
        save_json(str(gen_video_info_path), gen_video_info)
    print("视频片段生成完成，开始合成最终视频...")


    merged_video_path = output_path_dir / "merged_video.mp4"
    if not merged_video_path.exists():
        merge_all_videos(gen_video_info['video_part_info'], output_path=str(merged_video_path))
    print(f"最终视频合成完成,开始添加背景音乐...")

    final_video_path = output_path_dir / "final_video.mp4"
    add_bgm_to_video(str(merged_video_path), bgm_path, str(final_video_path))
    return str(final_video_path.resolve())


if __name__ == '__main__':
    question_id = "1896269793218242192"

    video_info_file = f"{question_id}/zhihu_answers_{question_id}_video_info_op.json"

    gen_video_by_video_info(video_info_file)