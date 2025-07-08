import scenedetect
import os
import csv  # 导入 csv 模块用于写入时间戳

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager  # <-- 1. 导入 StatsManager
from scenedetect.detectors import ContentDetector


def find_and_split_scenes(video_path, output_dir='videos', stats_file_prefix=''):
    """
    检测视频中的场景，分割视频，并存储精确的时间戳。

    参数:
    video_path (str): 输入视频文件的路径。
    output_dir (str): 分割后视频的输出目录。
    stats_file_prefix (str): 统计数据（时间戳）CSV文件的前缀。如果为空，则使用视频文件名。
    """
    # 创建一个 VideoManager 来管理视频文件
    video_manager = VideoManager([video_path])

    # 创建一个 StatsManager 来保存每个场景的详细统计信息，这是获取精确时间戳的关键
    stats_manager = StatsManager()

    # 创建一个 SceneManager，并将 StatsManager 传递给它
    scene_manager = SceneManager(stats_manager=stats_manager)

    # 添加内容检测器
    scene_manager.add_detector(ContentDetector(threshold=50, min_scene_len=25))

    try:
        # 设置 VideoManager 的属性
        base_timecode = video_manager.get_base_timecode()
        video_manager.set_downscale_factor()
        video_manager.start()

        # 在 SceneManager 中执行场景检测
        print(f'正在分析视频 {video_path}...')
        scene_manager.detect_scenes(frame_source=video_manager)

        # 获取检测到的场景列表 (包含开始和结束的时间码)
        scene_list = scene_manager.get_scene_list(base_timecode)

        print(f'成功检测到 {len(scene_list)} 个场景/片段。')

        # <-- 2. 保存精确时间戳到 CSV 文件
        if scene_list:
            # 如果未指定前缀，则使用视频文件名作为前缀
            if not stats_file_prefix:
                stats_file_prefix = os.path.splitext(os.path.basename(video_path))[0]

            stats_csv_path = os.path.join(output_dir, f'{stats_file_prefix}_timestamps.csv')

            print(f'正在将精确时间戳保存到 {stats_csv_path}...')

            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 写入 CSV 文件
            with open(stats_csv_path, 'w', newline='') as csv_file:
                # 使用 PySceneDetect 导出的场景列表来写入
                # 每个 scene 是一个元组 (开始时间码, 结束时间码)
                # 时间码对象有 .get_seconds() 和 .get_frames() 方法
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([
                    'Scene Number',
                    'Start Time (seconds)', 'End Time (seconds)', 'Duration (seconds)',
                    'Start Frame', 'End Frame', 'Duration (frames)',
                    'Start Timecode', 'End Timecode'
                ])
                for i, scene in enumerate(scene_list):
                    start_time, end_time = scene
                    csv_writer.writerow([
                        i + 1,
                        start_time.get_seconds(), end_time.get_seconds(), (end_time - start_time).get_seconds(),
                        start_time.get_frames(), end_time.get_frames(),
                        (end_time.get_frames() - start_time.get_frames()),
                        start_time.get_timecode(), end_time.get_timecode()
                    ])
            print('时间戳保存完毕！')

        # 如果检测到了场景，就进行分割
        if scene_list:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            print(f'开始分割视频，输出到 ./{output_dir} 目录...')
            scenedetect.split_video_ffmpeg(
                video_path,
                scene_list,
                output_dir=output_dir,
                show_progress=True
            )
            print('分割完成！')

    finally:
        video_manager.release()


# --- 主程序入口 ---
if __name__ == '__main__':
    # 把这里换成你的视频文件路径
    my_video_path = 'test.mp4'
    # 指定输出目录名
    output_directory = 'videos'

    find_and_split_scenes(my_video_path, output_dir=output_directory)