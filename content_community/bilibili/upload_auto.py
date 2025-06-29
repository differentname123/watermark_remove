import json
import os

# 假设你的上传函数在这里
from content_community.bilibili.bilibili_uploader import upload_to_bilibili

# --- 文件路径常量 ---
METADATA_FILE = '../../LLM/TikTokDownloader/metadata_cache.json'
# 使用 JSON 文件来记录已成功投稿的 video_id
UPLOADED_LOG_FILE = 'uploaded_videos.json'


def get_best_plan_by_potential(data: dict) -> dict:
    # 这个函数保持不变
    best_plan = None
    highest_score = float('-inf')

    for plan_name, plan_info in data.items():
        if not isinstance(plan_info, dict):
            continue
        score = plan_info.get("增长潜力", {}).get("爆款潜力指数", 0)
        if score > highest_score:
            highest_score = score
            best_plan = plan_info

    return best_plan


def load_uploaded_ids_from_json(filepath: str) -> set:
    """从 JSON 文件加载已上传的 video_id 列表。"""
    if not os.path.exists(filepath):
        return set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # 文件可能是空的，直接 load 会报错
            content = f.read()
            if not content:
                return set()
            ids_list = json.loads(content)
            return set(str(item) for item in ids_list)  # 转换为 set 并确保是字符串
    except (json.JSONDecodeError, IOError) as e:
        print(f"警告：无法读取或解析JSON记录文件 {filepath}。将创建一个新的记录。错误: {e}")
        return set()


def save_uploaded_ids_to_json(ids_set: set, filepath: str):
    """将 video_id 集合保存到 JSON 文件。"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # 将 set 转换为 list 以便 JSON 序列化
            json.dump(list(ids_set), f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"错误：无法写入记录到JSON文件 {filepath}: {e}")


def auto_upload():
    # 1. 从 JSON 文件加载已投稿的 video_id
    uploaded_ids = load_uploaded_ids_from_json(UPLOADED_LOG_FILE)
    print(f"已加载 {len(uploaded_ids)} 条已投稿记录。")

    # 2. 读取元数据文件
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata_cache = json.load(f)
    except FileNotFoundError:
        print(f"错误：元数据文件未找到 -> {METADATA_FILE}")
        return
    except json.JSONDecodeError:
        print(f"错误：元数据文件 JSON 格式有误 -> {METADATA_FILE}")
        return

    # 3. 遍历视频，进行投稿
    new_uploads_made = False  # 标记是否有新的上传成功
    for key, value in metadata_cache.items():
        metadata = value.get('metadata')
        if not metadata:
            print(f"跳过 {key}：缺少 'metadata' 字段。")
            continue
        video_id = metadata[0].get('id')
        if not video_id:
            print(f"跳过 {key}：'metadata' 中缺少 'id' 字段。")
            continue

        # 检查是否已投稿
        if str(video_id) in uploaded_ids:
            print(f"跳过 {key} (ID: {video_id})：此视频已投稿。") # 可以取消注释以获得更详细的输出
            continue

        video_path = value.get('video_path')
        if not video_path or not os.path.exists(video_path):
            print(f"跳过 {key} (ID: {video_id})：视频文件路径不存在或未提供 -> {video_path}")
            continue

        if 'best_scheme' in value and value['best_scheme']:
            best_scheme = value['best_scheme']
        else:
            title_schemes = value.get('title_schemes', {})
            best_scheme = get_best_plan_by_potential(title_schemes)

        if not best_scheme:
            print(f"跳过 {key} (ID: {video_id})：未能找到合适的投稿方案。")
            continue

        cover_path = best_scheme.get('封面', {}).get('图片路径', 'default_cover.jpg')
        title = best_scheme.get('标题', '欢迎来看我的视频！')
        description_json = best_scheme.get('简介', {})
        description = ""
        if isinstance(description_json, dict):
            description = "\n".join(description_json.values())
        elif isinstance(description_json, str):
            description = description_json

        tags = best_scheme.get('标签', ['AI修复', '视频剪辑', '有趣', '科技', '日常生活'])
        tags_str = ','.join(tags) if isinstance(tags, list) else tags

        print(f"准备投稿视频 (ID: {video_id})，标题：《{title}》")
        try:
            result = upload_to_bilibili(
                video_path=video_path,
                cover_path=cover_path,
                title=title,
                description=description,
                tags=tags_str,
            )
        except Exception as e:
            print(f"❌ 投稿失败：调用 upload_to_bilibili 时发生异常 -> {e}")
            continue

        if result and result.get('aid') and result.get('bvid'):
            print(f"🎉 投稿成功！")
            print(f"   - 标题: {title}")
            print(f"   - AID: {result['aid']}")
            print(f"   - BVID: {result['bvid']}")

            # 将新的 video_id 添加到集合中
            uploaded_ids.add(str(video_id))
            new_uploads_made = True  # 标记发生了成功的上传
        else:
            error_msg = result.get('message', '未知错误') if isinstance(result, dict) else str(result)
            print(f"❌ 投稿失败: {key} (ID: {video_id})。上传接口返回: {error_msg}")

    # 4. 如果有新的视频成功上传，则更新JSON文件
    if new_uploads_made:
        print(f"\n正在更新投稿记录文件: {UPLOADED_LOG_FILE}")
        save_uploaded_ids_to_json(uploaded_ids, UPLOADED_LOG_FILE)
        print("记录文件更新完毕。")

    print("\n所有任务处理完毕。")


if __name__ == "__main__":
    auto_upload()