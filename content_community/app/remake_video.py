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
from LLM.gemini import get_llm_content_gemini_flash_video
from common_utils.common_utils import string_to_object
from common_utils.split_scenes import find_and_split_scenes
from common_utils.video_utils import cover_video_area_gently

import json


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
    return result

def cover_subtitle(video_path):
    """
    覆盖视频中的字幕
    """
    output_path = video_path.replace('.mp4', '_covered.mp4')
    top_left = (62, 504)
    bottom_right = (1277, 614)
    cover_video_area_gently(
        video_path=video_path,
        output_path=output_path,
        top_left=top_left,
        bottom_right=bottom_right,
        mode='blur',
        strength=50  # 模糊强度，可以调整
    )
    return output_path



if __name__ == '__main__':
    # 把这里换成你的视频文件路径
    my_video_path = 'test.mp4'
    owner_speech_list = get_owner_speech(my_video_path)
    # 指定输出目录名
    output_directory = 'videos'

    # find_and_split_scenes(my_video_path, output_dir=output_directory)