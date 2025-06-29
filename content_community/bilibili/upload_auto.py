import json

from content_community.bilibili.bilibili_uploader import upload_to_bilibili

def get_best_plan_by_potential(data: dict) -> dict:
    best_plan = None
    highest_score = float('-inf')

    for plan_name, plan_info in data.items():
        score = plan_info.get("增长潜力", {}).get("爆款潜力指数", 0)
        if score > highest_score:
            highest_score = score
            best_plan = plan_info

    return best_plan



def auto_upload():

    with open('../../LLM/TikTokDownloader/metadata_cache.json', 'r', encoding='utf-8') as f:
        metadata_cache = json.load(f)

    for key, value in metadata_cache.items():
        video_path = value.get('video_path')
        if not video_path:
            print(f"跳过 {key}：没有找到视频路径")
            continue
        best_scheme = {}
        if 'best_scheme' in value:
            best_scheme = value['best_scheme']
        else:
            title_schemes = value.get('title_schemes', {})
            best_scheme = get_best_plan_by_potential(title_schemes)

        cover_path = best_scheme.get('封面', {}).get('图片路径', 'default_cover.jpg')
        title = best_scheme.get('标题', '欢迎来看我的视频！')
        description_json = best_scheme.get('简介')
        description = ""
        for key, value in description_json.items():
            description += f"{value}\n"
        tags = best_scheme.get('标签', 'AI修复,视频剪辑,有趣,科技,日常生活')
        tags = ', '.join(tags) if isinstance(tags, list) else tags
        result = upload_to_bilibili(
            video_path=video_path,
            cover_path=cover_path,
            title=title,
            description=description,
            tags=tags,
        )
        print(f"投稿成功！AID={result['aid']}, BVID={result['bvid']}")



if __name__ == "__main__":
    auto_upload()