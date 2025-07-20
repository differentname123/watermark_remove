import time
import requests
import os
import shutil
from typing import List, Tuple, Dict
import re  # <--- 1. 引入 re 模块

from common_utils.common_utils import get_config

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
API_KEY = get_config("gemini_api_key")
SEARCH_ENGINE_ID = "450cd42e2c0da4c9e"  # 这是你的搜索引擎 ID
SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
OUTPUT_DIR = 'google_images'

# --- 2. 添加 Headers，模拟浏览器 ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def search_google_images(query: str, count: int = 10) -> List[Tuple[str, str]]:
    """
    使用 Google Custom Search API 获取单个查询（可能包含多个关键词）的图片 URL。
    """
    if not API_KEY or "YOUR_GOOGLE_API_KEY" in API_KEY or not SEARCH_ENGINE_ID:
        print("错误: 请先在代码中填入 API_KEY 和 SEARCH_ENGINE_ID。")
        return []

    print(f"\n正在通过 Google 搜索: '{query}'...")
    image_data: List[Tuple[str, str]] = []

    for i in range(0, count, 10):
        num_this_page = min(10, count - i)
        start_index = i + 1
        params = {
            'key': API_KEY,
            'cx': SEARCH_ENGINE_ID,
            'q': query,
            'searchType': 'image',
            'num': num_this_page,
            'start': start_index,
            'gl': 'cn',
            'lr': 'lang_zh-CN',
            'safe': 'off'
        }
        try:
            response = requests.get(SEARCH_URL, params=params, headers=HEADERS)  # 添加 headers
            response.raise_for_status()
            items = response.json().get('items', [])
            for idx, item in enumerate(items):
                url = item.get('link')
                if not url:
                    continue

                # --- 3. 修正获取文件扩展名的方法 ---
                # 清理URL，去掉查询参数
                clean_url = url.split('?', 1)[0]
                # 使用正则表达式查找常见的图片扩展名
                match = re.search(r'\.(jpg|jpeg|png|gif|webp)', clean_url, re.IGNORECASE)
                if match:
                    ext = match.group(0)  # .jpg, .png etc.
                else:
                    ext = '.jpg'  # 如果找不到，则默认为 .jpg

                file_name = f"{query.replace(' ', '_')}_{start_index + idx}{ext}"
                image_data.append((file_name, url))
            if len(items) < num_this_page:
                break
        except requests.exceptions.RequestException as e:
            print(f"Google API 请求失败: {e}")
            break

    print(f"Google 成功找到 {len(image_data)} 张图片。")
    return image_data


def download_image(url: str, save_path: str) -> bool:
    if os.path.exists(save_path):
        print(f"  - 已存在，跳过: {os.path.basename(save_path)}")
        return True
    try:
        # --- 4. 在下载图片时也使用 Headers ---
        resp = requests.get(url, stream=True, timeout=20, headers=HEADERS)
        if resp.status_code == 200:
            with open(save_path, 'wb') as f:
                # 使用 shutil.copyfileobj 更高效
                shutil.copyfileobj(resp.raw, f)
            print(f"  ✔ 保存: {os.path.relpath(save_path)}")
            return True
        else:
            # 明确打印失败的状态码
            print(f"  ✘ 下载失败，状态码 {resp.status_code}，URL: {url}")
            return False
    except Exception as e:
        print(f"  ✘ 下载时发生异常: {e}")
        return False


def search_and_download(tasks: Dict[Tuple[str, ...], int]):
    """
    批量对多关键词组合进行搜索并下载。
    """
    # 建议不要每次都删除，除非你确实需要全新下载
    # if os.path.exists(OUTPUT_DIR):
    #     shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for kw_tuple, num in tasks.items():
        query = ' '.join(kw_tuple)
        folder_name = '_'.join(kw_tuple)
        save_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n任务: {query} (下载 {num} 张)")
        results = search_google_images(query, num)
        count = 0
        if not results:
            print(f"任务 '{query}' 未找到任何图片，跳过。")
            continue

        for fname, url in results:
            path = os.path.join(save_dir, fname)
            if download_image(url, path):
                count += 1
        print(f"任务 '{query}' 完成: 下载 {count}/{len(results)} 张图片。")  # 分母改为实际找到的数量更准确
        time.sleep(1)


def main():
    # 定义多关键词组合及下载数量
    tasks = {
        ('调查组', '会议', '专家', '政府工作组'): 4,
    }
    search_and_download(tasks)


if __name__ == '__main__':
    main()