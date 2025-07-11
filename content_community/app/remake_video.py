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
import traceback

import cv2

from LLM.gemini import get_llm_content_gemini_flash_video
from common_utils.common_utils import string_to_object, optimize_subtitle_timing, merge_time_segments, read_json, \
    save_json, fill_time_gaps, time_to_ms, find_file_by_name
from common_utils.ocr.paddle_ocr_utils import find_overall_subtitle_box, find_overall_subtitle_box_target_number
from common_utils.split_audio import separate_with_cli
from common_utils.split_scenes import find_and_split_scenes
from common_utils.tts.edge_tts_utils import generate_audio_and_get_duration_sync
from common_utils.tts.paddle_speech_demo import synthesize_and_get_duration
from common_utils.video_utils import cover_video_area_gently, add_subtitles_to_video, cover_video_area_simple, \
    re_edit_video_ffmpeg, extract_audio_from_video, cut_audio_segment
from paddlespeech.cli.tts.infer import TTSExecutor

import json

from common_utils.video_utils1 import redub_video_with_ffmpeg
from common_utils.video_utils2 import add_bgm_to_video


def get_owner_speech(video_path):
    """
    获取视频中的主人公语音片段。以及相应的语音
    """
    prompt = """
    你是一名专业的音频处理AI，任务是进行说话人识别、语音转写**和文案优化**。

    # 任务背景
    - 我是视频创作者，我将为你提供一份带有时间戳的语音转写初稿或音频文件。
    - 内容中混合了我的旁白、其他人的声音、以及外部视频片段的声音。

    # 任务目标
    1.  **识别主体**：在所有声音中，只识别并提取出属于“我”（视频创作者）的旁白部分。
    2.  **内容筛选**：完全忽略所有其他人声、背景音、以及非我本人说出的语句。
    3.  **精准对齐**：将我说的每一句旁白，都切分成一个符合自然语义的完整短句。每一句都必须带有精确到毫秒的起始和结束时间戳。
    4.  **验证校准**：如果给出了时间区间，请验证该区间是否准确对应我的声音，并进行必要的校准。
    **5.  文案润色 (新增目标):**
        -   **在完成上述步骤后，针对每一句识别出的原始旁白 (`text` 字段)，你需要生成一句新的文本。**
        -   **润色要求：**
            -   **保持原意**: 新句子的核心含义必须与原句完全一致。
            -   **长度严格一致**: **此为关键要求。** 新句子的长度（字数）**必须尽最大可能**与原句保持一致,或者少于原句子，最不希望大于原句子。这是为了确保优化后的文案能精准匹配原视频的时间轴和口型，因此请严格遵守此项规则。
            -   **整体通顺**: 所有润色后的新句子按顺序串联起来，也应能形成一篇通顺、连贯的文稿。

    # 输出要求
    - **格式**：最终结果必须是一个纯净、合法的 JSON 数组 (`Array of Objects`)。
    - **内容**：你的回答**必须且只能是**这个 JSON 数组本身，绝对不能包含任何解释性文字、注释、Markdown 标记（例如 ```json）或任何非 JSON 内容。
    - **结构**：数组中的每个对象代表我的一句旁白，包含以下**五个**字段：
        - `id`: (Number) 序号，从 1 开始递增。
        - `startTime`: (String) 开始时间，格式为 `HH:MM:SS.mmm`。
        - `endTime`: (String) 结束时间，格式为 `HH:MM:SS.mmm`。
        - `text`: (String) 旁白**原始**文本内容。
        - **`optimizedText`: (String) 经过润色后的新旁白文本，与原句意义相同、长度相近。**
    注意时间戳一定要是精确到毫秒的格式，且必须严格遵守 `HH:MM:SS.mmm` 的格式。
    # JSON 格式示例
    ```json
    [
      {
        "id": 1,
        "startTime": "00:00:03.125",
        "endTime": "00:00:05.890",
        "text": "欢迎来到我的视频。",
        "optimizedText": "欢迎来到我的频道。"
      },
      {
        "id": 2,
        "startTime": "00:00:07.500",
        "endTime": "00:00:10.000",
        "text": "今天我们来聊一个重要话题。",
        "optimizedText": "这次我们要谈一个核心要点。"
      }
    ]
    """
    base_name = os.path.basename(video_path)
    output_path = base_name.replace('.mp4', '_owner_speech.json')
    count = 0
    while True:
        count += 1
        if count > 3:
            print("重试次数超过3次，退出程序。")
            return []
        print("正在生成和优化字幕...")
        raw = get_llm_content_gemini_flash_video(prompt=prompt, video_path=video_path)
        result = string_to_object(raw)

        # 步骤 3: 优化字幕计时
        optimized_subtitles = optimize_subtitle_timing(result)

        if any(subtitle.get('duration', 0) < 0 for subtitle in optimized_subtitles):
            print("检测到无效的负数时长，将在2秒后重试...")
            continue  # 如果存在负数，则跳过本次循环的剩余部分，重新开始
        else:
            break  # 成功，跳出 while 循环

    # 步骤 6: 返回最终的、验证过的结果
    return optimized_subtitles


def cover_subtitle(video_path, output_path, top_left, bottom_right):
    """
    覆盖视频中的字幕
    """
    start_time = time.time()

    cover_video_area_simple(
        video_path=video_path,
        output_path=output_path,
        top_left=top_left,
        bottom_right=bottom_right
    )
    print(f"覆盖字幕区域完成，输出文件: {output_path} 耗时: {time.time() - start_time:.2f} 秒")

def gen_new_audio(optimized_subtitles,voice_name="zh-CN-YunjianNeural"):
    """
    生成语音文件，并更新字幕信息。

    默认使用第二种方式（如Azure TTS）进行语音合成。如果合成失败（返回时长为0.0），
    则自动切换到第一种备用方式（如本地TTS）重试。

    Args:
        optimized_subtitles (list): 包含字幕信息的列表，每个元素是一个字典。

    Returns:
        list: 更新了 'outputPath' 和 'trimmedDuration' 键的字幕列表。
    """
    output_dir = 'output_audio'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化备用语音合成引擎 (方式一)，只在需要时使用
    print("正在初始化备用语音合成引擎 (方式一)...")
    tts_engine_backup = TTSExecutor()

    for subtitle in optimized_subtitles:
        output_file = os.path.join(output_dir, f"{subtitle['id']}.wav")
        text_to_speak = subtitle['optimizedText']

        print(f"\n--- [字幕 {subtitle['id']}] 正在处理: '{text_to_speak}' ---")

        audio_length = generate_audio_and_get_duration_sync(
            text=text_to_speak,
            output_filename=output_file,
            voice_name=voice_name
        )

        # 检查方式二是否成功，如果不成功 (audio_length为0)，则切换到方式一
        if audio_length == 0.0:
            print(f"    [!] 默认方式生成失败，返回时长为 0.0。")
            print(f"    --> 切换到备用方式 (方式一) 重试...")

            audio_length = synthesize_and_get_duration(
                tts_executor=tts_engine_backup,
                text=text_to_speak,
                output_path=output_file
            )

        # 更新字幕信息
        subtitle['outputPath'] = output_file
        subtitle['trimmedDuration'] = audio_length

        # 打印最终结果
        if audio_length > 0.0:
            print(f"<-- [字幕 {subtitle['id']}] 生成成功！最终音频时长: {audio_length:.2f} 秒")
        else:
            print(f"<-- [字幕 {subtitle['id']}] 生成失败！两种方式都无法生成有效音频。")

    # 保存优化并更新后的字幕到文件
    print("\n所有音频处理完成，正在保存结果到 'optimized_subtitles.json'...")
    with open('optimized_subtitles.json', 'w', encoding='utf-8') as f:
        json.dump(optimized_subtitles, f, ensure_ascii=False, indent=4)
    print("文件保存成功！")

    # 直接返回内存中已更新的列表，无需重新读取文件
    return optimized_subtitles

def add_subtitle(input_video, subtitle_data, output_with_subtitles, bottom_margin, font_size, fixed_rect):
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
            bottom_margin=bottom_margin,
            fixed_rect=fixed_rect
        )

    except (FileNotFoundError, ValueError) as err:
        print(f"[主程序错误] 操作失败: {err}")
        print("\n[提示] 请确保：")
        print("1. `test.mp4` 文件存在于脚本相同目录下。")
        print("2. 你的系统中安装了 ffmpeg 并已添加到环境变量(PATH)。")
        print("3. 如果自动字体检测失败，请在代码中手动指定一个有效的中文字体路径。")


def adjust_subtitle_box(video_path: str, final_box: list[list[int, int]]):
    """
    调整字幕框左右边距为视频宽度的 10%，高度保持不变。

    参数:
        video_path: 视频文件路径
        final_box: 原始字幕框，格式 [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]

    返回:
        (top_left, bottom_right)：调整后的左上角和右下角坐标
    """
    # 1. 打开视频，获取分辨率
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 2. 计算新的左右边界x坐标，保留上下y坐标不变
    x_left = int(width * 0.1)  # 左侧距离视频宽度的 10%
    x_right = int(width * 0.9)  # 右侧距离视频宽度的 10%
    y_top = final_box[0][1]  # 原框上边 y 不变
    y_bottom = final_box[2][1]  # 原框下边 y 不变

    # 3. 构造新的 top_left 和 bottom_right
    top_left = [x_left, y_top]
    bottom_right = [x_right, y_bottom]

    return top_left, bottom_right, width, height


def gen_cut_suggestion(video_path):
    """
    生成剪辑的建议，交换场景顺序或者删除场景。
    """
    base_name = os.path.basename(video_path)
    output_path = base_name.replace('.mp4', '_cut_suggestion.json')
    try:
        if os.path.exists(output_path):
            print(f"检测到 {output_path} 已存在，直接读取...")
            with open(output_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return result
        scene_info_dict = find_and_split_scenes(video_path)
        if not scene_info_dict:
            print("未能成功获取视频场景信息。")
            return
        prompt = """# 角色
                    你是一位拥有十年经验的资深视频剪辑师和顶级社交媒体内容策略专家。你精通各种平台（如抖音、Bilibili、YouTube Shorts）的流量算法和用户心理，知道如何通过剪辑创造“黄金三秒”、提升完播率和互动率。
                    # 目标
                    你的核心目标是分析我提供的视频场景信息，并输出一个最优化的剪辑方案，旨在最大化视频的**观众吸引力、叙事流畅性、信息价值和传播潜力**。所有决策都必须以“让最终视频效果更好”为唯一标准。如果原始顺序已是最佳，则保持原样。
                    # 任务指令
                    1.  **全面分析**：基于提供的视频，深入理解整个视频的核心主题、叙事结构和关键信息点。
                    2.  **逐一评估**：结合“原始场景分割”，独立评估每个场景的作用和质量。评估维度包括：
                        *   **信息密度**：该场景是否传递了关键信息？
                        *   **视觉冲击力**：画面是否吸引人？
                        *   **情绪价值**：该场景能否引发观众的情绪（好奇、共鸣、兴奋等）？
                        *   **叙事功能**：它在故事中扮演什么角色（开端、发展、高潮、结尾、铺垫、转折）？
                        *   **冗余性**：该场景是否多余、拖沓或可被替代？
                    3.  **策略决策**：基于以上评估，构建最终的剪辑方案。你可以执行以下操作：
                        *   **保留 (Keep)**：当场景质量高且位置合适时。
                        *   **重排 (Reorder)**：调整场景顺序以优化叙事节奏或将最精彩的部分前置（例如，创建钩子）。
                        *   **删除 (Delete)**：移除内容冗余、质量低下或对主线故事无益的场景。
                    4.  **生成最终方案**：将你的决策结果以纯JSON格式输出。
                    
                    # 输出要求
                    *   **严格的JSON格式**：你的输出必须是**一个完整且格式正确的JSON对象**，不能包含任何JSON格式之外的标记、注释、代码块标识（如 ```json ... ```）或任何解释性文本。
                    *   **内容结构**：JSON对象必须包含以下三个顶级键：`overall_strategy`, `final_cut_sequence`, `deleted_scenes`。
                    
                    ---
                    ### JSON输出格式定义与示例
                    
                    ```json
                    {
                      "overall_strategy": "将成品展示（场景4）提前至开头作为钩子，吸引用户停留。删除了冗长的备料介绍（场景2），并整合了核心烹饪步骤，使节奏更紧凑，重点突出。",
                      "final_cut_sequence": [
                        {
                          "scene_id": "场景4",
                          "original_start_time": "00:02:19.827",
                          "original_end_time": "00:04:17.194",
                          "new_sequence_index": 1,
                          "reasoning": "作为视频钩子，快速展示最终成果，引发观众好奇心。"
                        },
                        {
                          "scene_id": "场景1",
                          "original_start_time": "00:00:00.000",
                          "original_end_time": "00:00:02.188",
                          "new_sequence_index": 2,
                          "reasoning": "简短的开场白，承接钩子，引入主题。"
                        },
                        {
                          "scene_id": "场景3",
                          "original_start_time": "00:00:53.328",
                          "original_end_time": "00:02:19.827",
                          "new_sequence_index": 3,
                          "reasoning": "核心内容，展示了关键的制作过程，保留以确保信息完整性。"
                        }
                      ],
                      "deleted_scenes": [
                        {
                          "scene_id": "场景2",
                          "original_start_time": "00:00:02.188",
                          "original_end_time": "00:00:53.328",
                          "reasoning": "此场景为详细的备料过程，节奏过于缓慢且信息密度低，删除可以使视频更紧凑，直接进入核心制作环节。"
                        }
                      ]
                    }
                    **原始场景分割信息如下**:
        """
        prompt = f"{prompt}\n{scene_info_dict}"
        raw = get_llm_content_gemini_flash_video(prompt=prompt, video_path=video_path)
        result = string_to_object(raw)
        return result
    except Exception as e:
        traceback.print_exc()
        return None

def auto_cut(video_path, all_info, output_path):
    """
    尝试进行场景的切换或者删除部分场景
    """
    if 'cut_suggestion_info' in all_info:
        cut_suggestion_info = all_info['cut_suggestion_info']
        print(f"检测到 {len(cut_suggestion_info.get('final_cut_sequence', []))} 个自动切割建议，直接使用...")
    else:
        cut_suggestion_info = gen_cut_suggestion(video_path)
        all_info['cut_suggestion_info'] = cut_suggestion_info

    if not cut_suggestion_info:
        pass
    final_cut_sequence = cut_suggestion_info.get('final_cut_sequence', [])
    merged_list = merge_time_segments(final_cut_sequence)
    re_edit_video_ffmpeg(video_path, merged_list, output_path=output_path)

    return cut_suggestion_info



def add_origin_audio(video_path, owner_speech_with_audio_list):
    """
    补充原来的声音，因为有些时候视频中引用了其他人的声音，现在需要保留下来
    """
    new_owner_speech_with_audio_list = fill_time_gaps(owner_speech_with_audio_list)
    if len(new_owner_speech_with_audio_list) > len(owner_speech_with_audio_list):
        origin_audio_path = video_path.replace('.mp4', '_origin_audio.wav')
        # 说明新增了片段，需要进行处理
        extract_audio_from_video(video_path, origin_audio_path)
        separate_with_cli(origin_audio_path, output_dir='origin_audio', two_stems=True)
        vocals_path = find_file_by_name('origin_audio', 'vocals.wav')
        if not vocals_path:
            vocals_path = origin_audio_path
            print("未找到分离的原始音频，使用原始音频作为补充。")

        for speech in new_owner_speech_with_audio_list:
            text = speech['text']
            if "[无声]" == text:
                speech_id = speech['id']
                audio_path = f'origin_audio/{speech_id}_origin.wav'
                startTime = speech['startTime']
                endTime = speech['endTime']
                start_time_s = time_to_ms(startTime) / 1000
                end_time_s = time_to_ms(endTime) / 1000
                cut_audio_segment(vocals_path, start_time_s, end_time_s, audio_path)
                if os.path.exists(audio_path):
                    speech['outputPath'] = audio_path
                print(f"已将无声片段 {speech_id} 的音频补充为原始音频: {audio_path}")


    return new_owner_speech_with_audio_list

def remake_video(video_path):
    """
    重制视频
    """
    base_name = os.path.basename(video_path)
    all_info_json_path = base_name.replace('.mp4', '_all_info.json')
    all_info = read_json(all_info_json_path)

    # 获取主人公语音片段
    if all_info and 'owner_speech' in all_info:
        owner_speech_list = all_info['owner_speech']
        print(f"检测到 {len(owner_speech_list)} 个主人公语音片段，直接使用...")
    else:
        owner_speech_list = get_owner_speech(video_path)
        all_info['owner_speech'] = owner_speech_list
        save_json(all_info_json_path, all_info)

    # 获取字幕框
    if 'final_subtitle_box' in all_info:
        final_box = all_info['final_subtitle_box']
        print(f"检测到 {final_box} 已存在的字幕框，直接使用...")
    else:
        final_box = find_overall_subtitle_box(video_path)
        all_info['final_subtitle_box'] = final_box
    top_left, bottom_right, vid_w, vid_h = adjust_subtitle_box(video_path, final_box)

    # 覆盖字幕区域
    covered_video_path = video_path.replace('.mp4', '_covered.mp4')
    if os.path.exists(covered_video_path):
        print(f"检测到 {covered_video_path} 已存在，直接使用...")
    else:
        print(f"正在覆盖字幕区域，输出文件: {covered_video_path}...")
        cover_subtitle(video_path, covered_video_path, top_left, bottom_right)


    # 增加新的文案和字幕
    add_subtitle_output_path = covered_video_path.replace('.mp4', '_with_subtitles.mp4')
    if os.path.exists(add_subtitle_output_path):
        print(f"检测到 {add_subtitle_output_path} 已存在，直接使用...")
    else:
        font_size = bottom_right[1] - top_left[1]
        font_size = int(font_size * 0.8)
        bottom_margin = vid_h - bottom_right[1] + int(int(bottom_right[1] - top_left[1]) * 0.1)
        add_subtitle(covered_video_path, owner_speech_list, add_subtitle_output_path, bottom_margin=bottom_margin, font_size=font_size, fixed_rect=[top_left, bottom_right])

    # add_subtitle_output_path = 'test_covered_with_subtitles.mp4'
    # 生成新的音频并且配上新的声音
    if 'new_owner_speech_with_audio_list' in all_info:
        new_owner_speech_with_audio_list = all_info['new_owner_speech_with_audio_list']
        print(f"检测到 {len(new_owner_speech_with_audio_list)} 个优化后的字幕，直接使用...")
    else:
        owner_speech_wiht_audio_list = gen_new_audio(owner_speech_list)
        new_owner_speech_with_audio_list = add_origin_audio(video_path, owner_speech_wiht_audio_list)
        all_info['new_owner_speech_with_audio_list'] = new_owner_speech_with_audio_list
        save_json(all_info_json_path, all_info)
    redub_output_file_path = add_subtitle_output_path.replace('.mp4', '_redub.mp4')
    # 使用ffmpeg重制视频
    # 过滤掉new_owner_speech_with_audio_list中text为"[无声]"的片段
    new_owner_speech_with_audio_list = [speech for speech in new_owner_speech_with_audio_list if speech['text'] !="[无声]"]
    redub_video_with_ffmpeg(add_subtitle_output_path, new_owner_speech_with_audio_list, output_path=redub_output_file_path)

    # # redub_output_file_path = 'test_covered_with_subtitles_redub.mp4'
    # # 自动切割视频，删除场景或者交换场景顺序
    # auto_cut_output_path = redub_output_file_path.replace('.mp4', '_auto_cut.mp4')
    # if os.path.exists(auto_cut_output_path):
    #     print(f"检测到 {auto_cut_output_path} 已存在，直接使用...")
    # else:
    #     print(f"正在自动切割视频，输出文件: {auto_cut_output_path}...")
    #     auto_cut(redub_output_file_path, all_info, auto_cut_output_path)
    #     save_json(all_info_json_path, all_info)
    #
    #
    # bgm_file = "background_music.mp3"
    # output_file = auto_cut_output_path.replace('.mp4', '_with_bgm.mp4')
    # add_bgm_to_video(redub_output_file_path, bgm_file, output_file)



if __name__ == '__main__':
    remake_video('test.mp4')
