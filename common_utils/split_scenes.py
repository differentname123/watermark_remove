import json

import scenedetect
import os
import pprint

from scenedetect import FrameTimecode
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from common_utils.image_utils import save_frames_around_timestamp


def find_and_split_scenes(
        video_path,
        high_threshold=50,
        min_scene_len=25,
        max_scenes=20,
        step=10,
        max_threshold=100
):
    """
    使用两阶段检测法，精确地分割视频场景。

    首先使用动态调整的高阈值进行粗略分割，以控制场景总数。
    然后使用低阈值进行精细检测，并用其结果对粗略场景的边界进行校正，
    以提高分割点的准确性。

    参数:
    video_path (str): 输入视频文件路径。
    high_threshold (int): 初始的高检测阈值。
    min_scene_len (int): 最小场景时长（帧）。
    max_scenes (int): 粗分割时允许的最大场景数量。
    step (int): 当场景过多时，每次调整高阈值的步长。
    max_threshold (int): 高阈值的上限。

    返回:
    dict: 经过精炼校正的场景信息字典，键为 "场景X", 值为 (start_timecode, end_timecode)。
    """
    current_high_threshold = high_threshold
    refined_scene_info = {}
    coarse_scene_info = {}
    fine_scene_info = {}
    video_manager = VideoManager([video_path])
    try:
        base_timecode = video_manager.get_base_timecode()
        video_manager.set_downscale_factor()
        video_manager.start()

        while current_high_threshold <= max_threshold:
            # --- 阶段 1: 使用高阈值进行粗略分割 ---
            print("-" * 80)
            print(f"阶段 1: 使用高阈值 {current_high_threshold} 进行粗略检测...")

            stats_manager_coarse = StatsManager()
            scene_manager_coarse = SceneManager(stats_manager=stats_manager_coarse)
            scene_manager_coarse.add_detector(
                ContentDetector(threshold=current_high_threshold, min_scene_len=min_scene_len)
            )

            # 检测场景
            scene_manager_coarse.detect_scenes(frame_source=video_manager)
            coarse_scene_list = scene_manager_coarse.get_scene_list(base_timecode)
            num_coarse_scenes = len(coarse_scene_list)
            print(f"检测到 {num_coarse_scenes} 个粗略场景。")

            # 如果场景数量满足条件，则进入精炼阶段
            if 0 < num_coarse_scenes <= max_scenes:
                print(f"场景数量 {num_coarse_scenes} 在目标范围 (1, {max_scenes}] 内，开始进行边界精炼。")

                # --- 阶段 2: 使用低阈值获取精细候选切点 ---
                low_threshold = current_high_threshold - 10
                print("-" * 80)
                print(f"阶段 2: 使用低阈值 {low_threshold:.2f} 获取精细候选切点...")
                coarse_scene_info = {
                    f"场景{i + 1}": (s.get_timecode(), e.get_timecode())
                    for i, (s, e) in enumerate(coarse_scene_list)
                }
                # !!! 关键步骤: 重置视频管理器到视频开头，以便重新检测
                video_manager.reset()

                stats_manager_fine = StatsManager()
                scene_manager_fine = SceneManager(stats_manager=stats_manager_fine)
                scene_manager_fine.add_detector(
                    ContentDetector(threshold=low_threshold, min_scene_len=min_scene_len)
                )

                # 重新检测
                scene_manager_fine.detect_scenes(frame_source=video_manager)
                fine_scene_list = scene_manager_fine.get_scene_list(base_timecode)
                print(f"检测到 {len(fine_scene_list)} 个精细候选场景。")
                fine_scene_info = {
                    f"场景{i+1}": (s.get_timecode(), e.get_timecode())
                    for i, (s, e) in enumerate(fine_scene_list)
                }
                # --- 阶段 3: 边界对齐与校正 ---
                print("-" * 80)
                print("阶段 3: 对齐粗场景边界到最近的精细切点...")

                # 提取所有粗略切点和精细切点的时间码 (FrameTimecode对象)
                coarse_cut_points = [scene[0] for scene in coarse_scene_list]
                fine_cut_points = [scene[0] for scene in fine_scene_list]

                if not fine_cut_points:
                    print("警告: 低阈值未检测到任何切点，将使用粗略分割结果。")
                    refined_scene_list = coarse_scene_list
                else:
                    refined_cut_points = [coarse_cut_points[0]]  # 第一个切点（视频开头）保持不变

                    # 对每个粗略切点（除了第一个），找到一个最接近的精细切点
                    for coarse_cut in coarse_cut_points[1:]:
                        closest_fine_cut = min(
                            fine_cut_points,
                            key=lambda fine_cut: abs(fine_cut.get_frames() - coarse_cut.get_frames())
                        )
                        refined_cut_points.append(closest_fine_cut)
                        print(f"  粗切点 {coarse_cut.get_timecode()} -> 校正为 {closest_fine_cut.get_timecode()} 差值 {abs(closest_fine_cut.get_frames() - coarse_cut.get_frames())} 帧")

                    # 使用校正后的切点列表重新构建场景列表
                    refined_scene_list = []
                    for i in range(len(refined_cut_points) - 1):
                        s, e = refined_cut_points[i], refined_cut_points[i + 1]
                        if s.get_frames() < e.get_frames():
                            refined_scene_list.append((s, e))

                    # 添加最后一个场景，其结束时间为视频的实际结尾
                    last_scene_start = refined_cut_points[-1]
                    video_end_time = coarse_scene_list[-1][1]  # 使用粗分割的结尾作为视频总长
                    refined_scene_list.append((last_scene_start, video_end_time))

                # 生成最终结果字典
                for i, scene in enumerate(refined_scene_list):
                    start_time, end_time = scene
                    scene_key = f"场景{i + 1}"
                    refined_scene_info[scene_key] = (
                        start_time.get_timecode(), end_time.get_timecode()
                    )
                break  # 成功处理，跳出 while 循环

            elif num_coarse_scenes == 0:
                print("未检测到任何场景，请尝试降低初始阈值。")
                break

            else:
                print(
                    f"场景数 {num_coarse_scenes} 大于限制 {max_scenes}, 将阈值调整为 {current_high_threshold + step} 并重试...")
                current_high_threshold += step
                # !!! 关键步骤: 重置视频管理器以进行下一次高阈值尝试
                video_manager.reset()

        else:  # while 循环正常结束 (即达到 max_threshold)
            print(f"已达到最大阈值 {max_threshold}, 仍无法将场景数降至 {max_scenes} 以下。")
            print("将使用最后一次检测到的粗略结果。")
            # 在这种情况下，也生成字典返回
            for i, scene in enumerate(coarse_scene_list):
                start_time, end_time = scene
                scene_key = f"场景{i + 1}"
                refined_scene_info[scene_key] = (
                    start_time.get_timecode(), end_time.get_timecode()
                )

    finally:
        video_manager.release()

    print("-" * 80)
    print("=== 粗场景 ===");   pprint.pprint(coarse_scene_info)
    print("=== 精细场景 ==="); pprint.pprint(fine_scene_info)
    print("=== 最终精炼场景分割结果 ==="); pprint.pprint(refined_scene_info)
    return refined_scene_info

def split_scenes_json(video_path: str,
                      high_threshold: int = 50,
                      min_scene_len: int = 25):
    """
    简化版场景分割函数（返回 JSON 字符串）。
    参数:
        video_path: 视频路径
        high_threshold: 高阈值（用于粗略检测）
        min_scene_len: 最小场景长度（帧）
    返回:
        JSON 字符串，形如: {"场景1": ["00:00:00.000","00:00:10.000"], "场景2": [...]}
        如果发生错误或检测到 0/1 个场景，仍然保证返回至少 "场景1"。
    依赖:
        pip install scenedetect（可选，若未安装将返回一个默认单场景）
    """
    result = {}
    try:
        # 延迟导入，避免在没有依赖时抛出 ImportError 导致函数崩溃
        from scenedetect import VideoManager, SceneManager
        from scenedetect.stats_manager import StatsManager
        from scenedetect.detectors import ContentDetector

        video_manager = VideoManager([video_path])
        try:
            base_timecode = video_manager.get_base_timecode()
            video_manager.set_downscale_factor()
            video_manager.start()

            # 第一轮：高阈值粗略检测
            stats_mgr = StatsManager()
            scene_mgr = SceneManager(stats_manager=stats_mgr)
            scene_mgr.add_detector(ContentDetector(threshold=high_threshold,
                                                   min_scene_len=min_scene_len))
            scene_mgr.detect_scenes(frame_source=video_manager)
            scene_list = scene_mgr.get_scene_list(base_timecode)

            # 如果没检测到，尝试降低阈值再试一次（一次）
            if not scene_list:
                video_manager.reset()
                stats_mgr = StatsManager()
                scene_mgr = SceneManager(stats_manager=stats_mgr)
                low_thr = max(0, high_threshold - 10)
                scene_mgr.add_detector(ContentDetector(threshold=low_thr,
                                                       min_scene_len=min_scene_len))
                scene_mgr.detect_scenes(frame_source=video_manager)
                scene_list = scene_mgr.get_scene_list(base_timecode)

            # 如果仍然没有结果，或只有一个场景，也保证返回场景1
            if not scene_list:
                # 无法获得时码信息，返回默认单场景（可被前端/调用方识别为“检测失败”或空片段）
                result["场景1"] = ["00:00:00.000", "00:00:00.000"]
            else:
                # 组装结果（确保即使只有一个场景也返回）
                for i, (s, e) in enumerate(scene_list):
                    result[f"场景{i+1}"] = [s.get_timecode(), e.get_timecode()]

                # 如果只检测到 1 个场景且调用者希望强制至少一个场景，
                # 上面已经满足（返回场景1）
        finally:
            try:
                video_manager.release()
            except Exception:
                pass

    except Exception:
        # 任何导入错误或运行时错误，都返回单场景保证调用端不会崩溃
        result = {"场景1": ["00:00:00.000", "00:00:00.000"]}

    # 返回 JSON 字符串（非 ASCII 编码，便于中文键名）
    return result


# --- 主程序入口 ---
if __name__ == '__main__':
    # 把这里换成你的视频文件路径
    my_video_path = 'test2.mp4'

    # 运行带有精炼功能的场景分割
    scene_info_dict = find_and_split_scenes(
        my_video_path,
        high_threshold=50,  # 初始高阈值
        max_scenes=20,  # 期望的最大场景数
        min_scene_len=25,  # 最小场景长度（帧）
        step=5  # 阈值调整步长
    )
    print("\n场景信息字典已生成并打印。")
    for key, value in scene_info_dict.items():
        timestamp = value[1]
        save_frames_around_timestamp(my_video_path, timestamp, 3, str(os.path.join('scenes', key)))