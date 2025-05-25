# -- coding: utf-8 --
""":authors:
    zhuxiaohu
:create_date:
    2025/5/25 21:53
:last_date:
    2025/5/25 21:53
:description:
    
"""
import time

from LLM.doubao import detection_watermark
from inpainting.lama_inpainting import inpating_video


def run():
    start_time = time.time()
    video_name = "test1.mp4"
    clusters = detection_watermark(video_file=video_name, num_frames=10)
    box_list = [cluster["enclosing_box"] for cluster in clusters]

    inpating_video(video_name, box_list)
    print(f"{video_name} 处理时间: {time.time() - start_time:.2f} 秒")



if __name__ == "__main__":
    run()