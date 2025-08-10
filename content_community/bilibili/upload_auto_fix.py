import os

import requests
from http import HTTPStatus
from typing import List, Dict

from common_utils.common_utils import get_config, find_key_values, read_json, save_json
from content_community.bilibili.high_quality_hudong import find_video_by_bvid


def parse_cookie_string(cookie_str: str) -> Dict[str, str]:
    """
    将 "a=1; b=2; c=3" 形式的 Cookie 字符串解析成字典
    """
    cookies = {}
    for kv in cookie_str.split(';'):
        if '=' in kv:
            k, v = kv.strip().split('=', 1)
            cookies[k] = v
    return cookies

def fetch_all_archives(cookie_str: str) -> List[Dict]:
    """
    循环拉取 B 站未发布稿件列表，直到没有更多数据或遇到致命错误为止。
    如果某页返回 412，将记录并中断循环。
    """
    # 1. 创建 Session 并设置 Cookie
    session = requests.Session()
    session.cookies.update(parse_cookie_string(cookie_str))

    # 2. 填充通用请求头
    session.headers.update({
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
        'Origin': 'https://member.bilibili.com',
        'Referer': 'https://member.bilibili.com/platform/upload-manager/article?group=not_pubed&page=1',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'cors',
        'X-Requested-With': 'XMLHttpRequest'
    })

    base_url = 'https://member.bilibili.com/x/web/archives'
    pn = 1
    ps = 10
    all_items: List[Dict] = []
    try:
        while True:
            params = {
                'status': 'not_pubed',
                'pn': pn,
                'ps': ps,
                'coop': 1,
                'interactive': 1
            }
            try:
                resp = session.get(base_url, params=params, timeout=10)
                # 如果是 412，抛出 HTTPError，我们在 except 里捕获后中断
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if resp.status_code == HTTPStatus.PRECONDITION_FAILED:
                    print(f'第 {pn} 页返回 412 (Precondition Failed)，可能是缺少必要头部或 Cookie 已失效。中断循环。')
                else:
                    print(f'第 {pn} 页请求失败：{resp.status_code} {resp.reason}')
                break
            except requests.exceptions.RequestException as e:
                print(f'第 {pn} 页请求异常：{e}，中断循环。')
                break

            data = resp.json()
            archives = data.get('data', {}).get('arc_audits', [])
            print(f'第 {pn} 页，获取到 {len(archives)} 条')

            if not archives:
                print('已无更多数据，退出循环。')
                break

            all_items.extend(archives)
            pn += 1
    except Exception as e:
        print(f'发生异常：{e}，中断循环。')

    return all_items

if __name__ == '__main__':
    # 示例调用
    name_list = ['ruru', 'qiqi', 'jie', 'yan', 'cai']
    name_list = ['nana']

    all_items = []
    for name in name_list:
        COOKIE_STRING = get_config(f"{name}_bilibili_total_cookie")
        items = fetch_all_archives(COOKIE_STRING)
        all_items.extend(items)
    bvid_list = find_key_values(all_items, 'bvid')
    bvid_list = list(set(bvid_list))
    print(f'共获取 {len(bvid_list)} 个 BVID')
    metadata_cache_with_uploads = {}
    # 指定要扫描的目录
    root_dir = '../../LLM/TikTokDownloader'
    # 用于保存符合条件的文件路径
    matched_files = []
    # 遍历目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if 'metadata_cache_with_uploads' in filename:
                # 拼接完整路径
                full_path = os.path.join(dirpath, filename)
                matched_files.append(full_path)
    # 输出所有匹配的文件
    for file_path in matched_files:
        metadata_cache_with_uploads.update(read_json(file_path))

    fix_metadata_cache_with_uploads_path = '../../LLM/TikTokDownloader/metadata_cache_with_uploads_fix.json'
    fix_metadata_cache_with_uploads = {}
    count = 0
    for bvid in bvid_list:
        target_value = find_video_by_bvid(bvid, metadata_cache_with_uploads) or {}
        if target_value:
            id = target_value.get('metadata')[0].get('id')
            fix_metadata_cache_with_uploads[id] = target_value
            count+= 1
    save_json(fix_metadata_cache_with_uploads_path, fix_metadata_cache_with_uploads)
    print(f'共修复 {count} 条数据')
