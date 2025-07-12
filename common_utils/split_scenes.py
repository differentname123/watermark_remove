import scenedetect
import os
import csv
import pprint  # <-- 1. 导入 pprint 模块，用于美观地打印字典

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector


def find_and_split_scenes(
        video_path,
        output_dir='videos',
        stats_file_prefix='',
        threshold=50,
        min_scene_len=25,
        max_scenes=20,
        step=10,
        max_threshold=100
):
    """
    检测视频中的场景，并自动调整阈值直到场景数量不超过 max_scenes 或达到最大阈值。

    参数:
    video_path (str): 输入视频文件路径。
    output_dir (str): 分割后视频的输出目录。
    stats_file_prefix (str): 统计数据（时间戳）CSV文件前缀。
    threshold (int): 初始检测阈值。
    min_scene_len (int): 最小场景时长（帧）。
    max_scenes (int): 最大允许的场景数量。
    step (int): 每次调整阈值的步长。
    max_threshold (int): 最大阈值上限。
    返回:
    dict: 场景信息字典，键为 "场景1", 值为 (start_timecode, end_timecode)。
    """
    current_threshold = threshold
    scene_info_dict = {}

    while current_threshold <= max_threshold:
        # 初始化管理器
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager=stats_manager)
        scene_manager.add_detector(
            ContentDetector(threshold=current_threshold, min_scene_len=min_scene_len)
        )

        try:
            base_timecode = video_manager.get_base_timecode()
            video_manager.set_downscale_factor()
            video_manager.start()

            print(f'使用阈值 {current_threshold} 分析视频 {video_path}...')
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list(base_timecode)
            num_scenes = len(scene_list)
            print(f'检测到 {num_scenes} 个场景。')

            # 如果场景数量满足条件，则跳出循环
            if num_scenes <= max_scenes:
                # 生成结果字典
                for i, scene in enumerate(scene_list):
                    start_time, end_time = scene
                    scene_key = f"场景{i + 1}"
                    scene_info_dict[scene_key] = (
                        start_time.get_timecode(), end_time.get_timecode()
                    )
                break
            else:
                # 增加阈值并重试
                print(f'场景数 {num_scenes} 大于限制 {max_scenes}, 将阈值调整为 {current_threshold + step} 并重试...')
                current_threshold += step
        finally:
            video_manager.release()
    else:
        print(f'已达到最大阈值 {max_threshold}, 仍检测到 {num_scenes} 个场景。')
        # 即便超过阈值，仍返回最后一次的结果字典

    pprint.pprint(scene_info_dict)
    return scene_info_dict


# --- 主程序入口 ---
if __name__ == '__main__':
    # 把这里换成你的视频文件路径
    my_video_path = '../content_community/app/test1_covered_with_subtitles_redub.mp4'
    # 指定输出目录名
    output_directory = 'videos'

    scene_info_dict = find_and_split_scenes(my_video_path)
    print("\n场景信息字典已生成并打印。")