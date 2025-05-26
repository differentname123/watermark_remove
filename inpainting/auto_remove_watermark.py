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

from LLM.doubao import detection_watermark_llm
from inpainting.lama_inpainting import inpating_video
from inpainting.watermark_detection_ocr import detection_watermark_ocr


def run():
    start_time = time.time()
    video_name = "test.mp4"
    clusters = detection_watermark_llm(video_file=video_name, num_frames=10)
    # clusters = detection_watermark_ocr(video_file=video_name, num_frames=10)

    box_list = [cluster["enclosing_box"] for cluster in clusters]

    inpating_video(video_name, box_list, max_frames=100)
    print(f"{video_name} 处理时间: {time.time() - start_time:.2f} 秒")



if __name__ == "__main__":
    run()