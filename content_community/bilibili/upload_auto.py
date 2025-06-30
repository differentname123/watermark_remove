import json
import os

from content_community.bilibili.bilibili_uploader import upload_to_bilibili

# --- 文件路径常量 ---
METADATA_FILE = '../../LLM/TikTokDownloader/metadata_cache.json'

def get_best_plan_by_potential(data: dict) -> dict:
    """根据爆款潜力指数选择最佳方案。"""
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


def auto_upload():
    """
    自动读取元数据、选择最佳方案并上传视频。
    如果视频已上传过，则跳过。
    上传成功后，将上传信息记录回元数据文件。
    """
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata_cache = json.load(f)
    except FileNotFoundError:
        print(f"错误：元数据文件未找到 -> {METADATA_FILE}")
        return
    except json.JSONDecodeError:
        print(f"错误：元数据文件 JSON 格式有误 -> {METADATA_FILE}")
        return

    new_uploads_made = False  # 标记是否有新的上传成功

    # 遍历所有视频元数据
    for key, value in metadata_cache.items():
        print("-" * 50)

        # 1. 检查是否已经投稿成功
        if 'upload_info' in value:
            print(f"✅ 跳过 {key}：该视频已于之前投稿成功。")
            continue

        metadata = value.get('metadata')
        if not metadata:
            print(f"⏭️ 跳过 {key}：缺少 'metadata' 字段。")
            continue
        video_id = metadata[0].get('id')
        if not video_id:
            print(f"⏭️ 跳过 {key}：'metadata' 中缺少 'id' 字段。")
            continue

        video_path = value.get('video_path')
        if not video_path or not os.path.exists(video_path):
            print(f"⏭️ 跳过 {key} (ID: {video_id})：视频文件路径不存在或未提供 -> {video_path}")
            continue

        # 2. 准备投稿所需信息
        if 'best_scheme' in value and value['best_scheme']:
            best_scheme = value['best_scheme']
        else:
            title_schemes = value.get('title_schemes', {})
            best_scheme = get_best_plan_by_potential(title_schemes)

        if not best_scheme:
            print(f"⏭️ 跳过 {key} (ID: {video_id})：未能找到合适的投稿方案。")
            continue

        abs_cover_path = metadata[0].get('abs_cover_path')
        if abs_cover_path and os.path.exists(abs_cover_path):
            cover_path = abs_cover_path
        else:
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
        dynamic = best_scheme.get('简介', {}).get('互动引导', '希望大家喜欢')

        # 将本次投稿所用的参数打包成一个字典
        upload_params = {
            'title': title,
            'description': description,
            'tags': tags_str,
            'dynamic': dynamic,
            'cover_path': cover_path,
            'video_path': video_path
        }

        print(f"🚀 准备投稿视频 (ID: {video_id})，标题：《{title}》")

        # 3. 执行投稿
        try:
            result = upload_to_bilibili(**upload_params)
        except Exception as e:
            print(f"❌ 投稿失败：调用 upload_to_bilibili 时发生异常 -> {e}")
            continue

        # 4. 处理投稿结果
        # B站成功投稿通常返回的code为0
        if result and result.get('aid') and result.get('bvid'):
            print(f"🎉 投稿成功！")
            print(f"   - 标题: {title}")
            print(f"   - AID: {result.get('aid')}")
            print(f"   - BVID: {result.get('bvid')}")

            # 创建 upload_info 字段并添加到当前视频的 value 中
            upload_info = {
                'upload_params': upload_params,
                'upload_result': result
            }
            metadata_cache[key]['upload_info'] = upload_info  # 直接修改内存中的字典

            new_uploads_made = True  # 标记发生了成功的上传
        else:
            error_msg = result.get('message', '未知错误') if isinstance(result, dict) else str(result)
            print(f"❌ 投稿失败: {key} (ID: {video_id})。上传接口返回: {error_msg}")

    # 5. 如果有新的视频成功上传，则将更新后的 metadata_cache 写回文件
    print("=" * 50)
    if new_uploads_made:
        print("\n检测到新的成功投稿，正在更新元数据文件...")
        try:
            with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(metadata_cache, f, indent=4, ensure_ascii=False)
            print(f"✅ 元数据文件 {METADATA_FILE} 更新成功。")
        except IOError as e:
            print(f"🔥 错误：无法写入更新后的元数据到文件 {METADATA_FILE}: {e}")
    else:
        print("\n本次运行没有新的成功投稿，元数据文件无需更新。")

    print("\n所有任务处理完毕。")


if __name__ == "__main__":
    auto_upload()