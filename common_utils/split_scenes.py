import scenedetect
import os
import csv
import pprint  # <-- 1. 导入 pprint 模块，用于美观地打印字典

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector


def find_and_split_scenes(video_path, output_dir='videos', stats_file_prefix='', threshold=50, min_scene_len=25):
    """
    检测视频中的场景，分割视频，存储精确的时间戳，并打印场景信息字典。

    参数:
    video_path (str): 输入视频文件的路径。
    output_dir (str): 分割后视频的输出目录。
    stats_file_prefix (str): 统计数据（时间戳）CSV文件的前缀。如果为空，则使用视频文件名。
    """
    # 创建一个 VideoManager 来管理视频文件
    video_manager = VideoManager([video_path])

    # 创建一个 StatsManager 来保存每个场景的详细统计信息
    stats_manager = StatsManager()

    # 创建一个 SceneManager
    scene_manager = SceneManager(stats_manager=stats_manager)

    # 添加内容检测器
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))

    try:
        # 设置 VideoManager 的属性
        base_timecode = video_manager.get_base_timecode()
        video_manager.set_downscale_factor()
        video_manager.start()

        # 在 SceneManager 中执行场景检测
        print(f'正在分析视频 {video_path}...')
        scene_manager.detect_scenes(frame_source=video_manager)

        # 获取检测到的场景列表 (包含开始和结束的时间码对象)
        scene_list = scene_manager.get_scene_list(base_timecode)

        print(f'成功检测到 {len(scene_list)} 个场景/片段。')

        # --------------------------------------------------------------------
        # <-- 2. 新增部分：创建并打印你需要的场景信息字典
        # --------------------------------------------------------------------
        if scene_list:
            scene_info_dict = {}
            print("\n" + "=" * 20)
            print("场景详细信息字典:")
            for i, scene in enumerate(scene_list):
                start_time, end_time = scene
                # 构建字典的键，例如 "场景1"
                scene_key = f"场景{i + 1}"
                # 构建值，即一个包含开始和结束精确时间码字符串的元组
                # .get_timecode() 方法返回 "HH:MM:SS.ms" 格式的字符串
                scene_info_dict[scene_key] = (start_time.get_timecode(), end_time.get_timecode())

            # 使用 pprint 美观地打印字典
            pprint.pprint(scene_info_dict)
            print("=" * 20 + "\n")
        return scene_info_dict
    finally:
        video_manager.release()
        return {}


# --- 主程序入口 ---
if __name__ == '__main__':
    # 把这里换成你的视频文件路径
    my_video_path = '../content_community/app/test1_covered_with_subtitles_redub.mp4'
    # 指定输出目录名
    output_directory = 'videos'

    scene_info_dict = find_and_split_scenes(my_video_path)
    print("\n场景信息字典已生成并打印。")