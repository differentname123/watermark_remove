import time
import requests
import os
import shutil
from typing import List, Tuple, Dict, Deque
import re
from collections import deque  # <--- 1. 引入 deque，一个高效的队列
from PIL import Image  # <--- 2. 引入 Pillow 库用于图片验证

from common_utils.common_utils import get_config

# --- 环境变量和全局常量 (保持不变) ---
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
API_KEY = get_config("gemini_api_key")
SEARCH_ENGINE_ID = "450cd42e2c0da4c9e"
SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
OUTPUT_DIR = 'google_images'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def search_google_images_page(query: str, start_index: int) -> List[Tuple[str, str]]:
    """
    <--- 3. 功能修改: 此函数现在只获取单页（最多10个）的搜索结果。
    这将帮助我们在主循环中按需获取新的URL。
    """
    if not API_KEY or "YOUR_GOOGLE_API_KEY" in API_KEY or not SEARCH_ENGINE_ID:
        print("错误: 请先在代码中填入 API_KEY 和 SEARCH_ENGINE_ID。")
        return []

    print(f"  > 正在向 Google 请求更多图片 (从第 {start_index} 张开始)...")
    image_data: List[Tuple[str, str]] = []
    params = {
        'key': API_KEY, 'cx': SEARCH_ENGINE_ID, 'q': query,
        'searchType': 'image', 'num': 10, 'start': start_index,
        'gl': 'cn', 'lr': 'lang_zh-CN', 'safe': 'off'
    }
    try:
        response = requests.get(SEARCH_URL, params=params, headers=HEADERS)
        response.raise_for_status()
        items = response.json().get('items', [])
        for idx, item in enumerate(items):
            url = item.get('link')
            if not url:
                continue

            clean_url = url.split('?', 1)[0]
            match = re.search(r'\.(jpg|jpeg|png|gif|webp|bmp)', clean_url, re.IGNORECASE)
            ext = match.group(0) if match else '.jpg'

            # 使用更独特的文件名，避免潜在冲突
            file_name = f"{query.replace(' ', '_')}_{start_index + idx}{ext}"
            image_data.append((file_name, url))

        if not items:
            print("  > Google API 未返回更多图片。")
        return image_data
    except requests.exceptions.RequestException as e:
        print(f"  > Google API 请求失败: {e}")
        return []


def download_and_validate_image(url: str, save_path: str) -> bool:
    """
    <--- 4. 全新重构的下载函数，增加了验证步骤。
    只有下载成功并且确认为有效图片时，才返回 True。
    """
    if os.path.exists(save_path):
        print(f"    - 已存在，跳过: {os.path.basename(save_path)}")
        return True  # 如果文件已存在，我们假设它是好的，计入成功计数

    try:
        resp = requests.get(url, stream=True, timeout=20, headers=HEADERS)
        if resp.status_code != 200:
            print(f"    ✘ 下载失败 (状态码 {resp.status_code}) URL: {url}")
            return False

        # <--- 5. 检查 Content-Type，进行初步过滤 ---
        content_type = resp.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            print(f"    ✘ 非图片类型 ({content_type})，跳过 URL: {url}")
            return False

        # 下载到临时文件
        temp_path = save_path + ".tmp"
        with open(temp_path, 'wb') as f:
            shutil.copyfileobj(resp.raw, f)

        # <--- 6. 核心验证: 使用 Pillow 确认图片是否有效 ---
        try:
            # 验证文件大小，过滤掉过小的文件（通常是错误页或追踪像素）
            if os.path.getsize(temp_path) < 2 * 1024:  # 小于 2KB 的很可疑
                print(f"    ✘ 文件太小 ({os.path.getsize(temp_path)} bytes)，可能无效，删除。")
                os.remove(temp_path)
                return False

            with Image.open(temp_path) as img:
                img.verify()  # 验证图片数据的完整性

            # 验证通过，将临时文件重命名为最终文件
            os.rename(temp_path, save_path)
            print(f"    ✔ 保存并验证成功: {os.path.relpath(save_path)}")
            return True
        except Exception as e:
            # 如果 Pillow 无法打开或验证，说明文件损坏或格式错误
            os.remove(temp_path)
            print(f"    ✘ 图片验证失败 (损坏或非图片格式)，已删除。错误: {e}")
            return False

    except Exception as e:
        print(f"    ✘ 下载过程中发生未知异常: {e}")
        # 如果临时文件存在，清理掉
        if os.path.exists(save_path + ".tmp"):
            os.remove(save_path + ".tmp")
        return False


def search_and_download(tasks: Dict[Tuple[str, ...], int]):
    """
    <--- 7. 主逻辑重构，实现持续下载直到满足数量。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for kw_tuple, target_count in tasks.items():
        query = ' '.join(kw_tuple)
        folder_name = '_'.join(kw_tuple)
        save_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n--- 任务开始: {query} (目标: 下载 {target_count} 张) ---")

        successful_downloads = 0
        urls_to_try: Deque[Tuple[str, str]] = deque()
        api_start_index = 1
        attempted_urls = set()  # 记录所有尝试过的URL，避免重复下载

        while successful_downloads < target_count:
            # 如果我们的URL队列空了，就去获取更多
            if not urls_to_try:
                # Google API 限制最多100个结果 (start_index 不能超过 91)
                if api_start_index > 91:
                    print(f"警告: 已达到 Google API 单次查询 100 张图片的上限。无法获取更多图片。")
                    break

                new_results = search_google_images_page(query, api_start_index)
                if not new_results:
                    # 如果API没有返回新结果，说明这个关键词的图片已经全部找完了
                    print("提示: Google 已无更多相关图片返回。")
                    break

                for fname, url in new_results:
                    if url not in attempted_urls:
                        urls_to_try.append((fname, url))
                        attempted_urls.add(url)

                api_start_index += 10  # 准备下一次API请求的起始位置

            # 如果队列中还是没有URL（可能新获取的都是重复的），则退出
            if not urls_to_try:
                print("提示: 获取到的新URL均为重复，停止任务。")
                break

            # 从队列中取出一个URL进行下载
            file_name, url = urls_to_try.popleft()
            save_path = os.path.join(save_dir, file_name)

            if download_and_validate_image(url, save_path):
                successful_downloads += 1
                print(f"  进度: {successful_downloads}/{target_count}")

        print(f"--- 任务 '{query}' 完成: 最终成功下载 {successful_downloads}/{target_count} 张有效图片。---\n")
        time.sleep(1)


def main():
    tasks = {
        ('调查组', '会议', '专家', '政府工作组'): 5,  # <--- 尝试一个大于10的数字来测试分页
    }
    search_and_download(tasks)


if __name__ == '__main__':
    main()