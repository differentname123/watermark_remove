import datetime
import multiprocessing
import os
import random
import re
import time
import traceback
from typing import Any, Dict, List, Optional
from multiprocessing import Pool

from LLM.gemini import get_llm_content
from common_utils.common_utils import read_json, get_config, save_json_safe, init_config, process_product_title
from common_utils.video_utils import create_enhanced_cover
from content_community.bilibili.add_good_comment import format_title, _initialize_commenters, _should_skip_video, \
    RETRY_INTERVAL_SECONDS, REDUCED_MAX_REPLIES_PER_RUN, EXISTING_REPLIES_THRESHOLD, DEFAULT_MAX_REPLIES_PER_RUN, \
    _is_rpid_in_comments
from content_community.bilibili.add_good_comment_kouling import search_goods
from content_community.bilibili.bili_utils import fetch_goods, update_short_url, list_selection_car_items, \
    get_bili_reply_text, del_bili_reply_text, set_bili_reply, set_bili_keys_reply, update_bili_user_sign
from content_community.bilibili.comment import BilibiliCommenter
from content_community.bilibili.get_comment import get_bilibili_comments
from content_community.bilibili.get_danmu import string_to_list
from content_community.bilibili.high_quality_hudong import find_video_by_bvid
from common_utils.common_utils import string_to_object, save_json_safe, read_json
from content_community.taobao.taobao_utils import add_to_favorites, creat_and_favorite

BASE_DIR = 'replay_info'

final_prompt = """
你是一位“置顶神评”工程的**首席架构师**，一位精通用户心理、专注“无痕转化”的顶级专家。你的唯一任务是：在收到短视频的JSON数据后，设计并产出1-2条具备病毒式传播潜力的置顶神评，实现“情绪共鸣最大化”与“资源引导率最大化”的双重目标。**严格遵循所有指令，最终仅返回一个纯净的JSON对象。**

#### **一、核心心法与铁律 (必须内化为本能)**

1.  **双重身份 (Dual Identity)**

      * **情绪嘴替 (Emotional Spokesperson)**：你的评论必须一击即中视频的唯一“情绪核”，替千万观众说出那句他们“心中有，口中无”的终极心声。
      * **价值整理者 (The Value Curator)**：你不是在推销，而是在为所有和自己一样“上头”的观众，**主动付出了时间和精力**，将视频背后最关键、最爽的核心价值**“整理成了一份宝藏”**。你的动机是利他的，姿态是“我为大家省时间”。


#### **二、输入规范 (保留)**

  * `titles`, `video_anlyse`, `comments`, `resources`

#### **三、四步创作流程 (SOP - 必须严格执行)**

**第一步：洞察与情绪锚定 (Insight & Anchoring)**

  * 快速整合所有输入，定位视频唯一的、最强烈的**“情绪核”**。

**第二步：资源匹配与价值提炼 (Resource Matching & Value Distillation)**

  * 基于“情绪核”，筛选出1-2个最契合的资源。
  * 为每个资源计算**“单点转化指数” `score`**，并为其提炼一个精准的**“价值包装词”**（从合规词库中选择）。

**第三步：神评架构与写作 (Comment Architecture & Writing)**

  * 评论必须是**两段式、单行无换行、总字数不超过90字**。

      * **[情绪共鸣体] (主体, 严格控制在 40-50 字)**

          * **目标**: 独立高赞，成为“情绪嘴替”。
          * **战术**: **细节洞察式** / **情感升华式** / **机智吐槽式**。

      * **[直白价值钩] (结尾, 严格控制在 15-30 字) - (核心升级)**

          * **目标**: 清晰告知你提供了什么价值，直接激发用户的获取欲。
          * **全新公式**: **“自然连接词 + 行动声明 + 价值包装”**
          * **1. 自然连接词**: `…看完` / `，于是` / `——` / `，顺手就`
          * **2. 行动声明 (关键)**: **`我整理了` / `我花时间盘了盘` / `我把...做成了`**
          * **3. 价值包装**: 落在第二步提炼的**“价值包装词”**上。
          * **全新范例**:
              * "...看完太上头了，**我把**视频里所有能复刻这种爽感的**捷径整理成了一份清单**。"
              * "...于是**我花了两晚**，**把**这种专治不服的快乐**做成了一个合集**。"

**第四步：自检与输出 (Final Check & Output)**

  * **内部自检**:
    1.  前半段能否独立高赞？
    2.  是否严格遵守了所有格式要求？
  * **最终输出**: 仅返回JSON。

#### **四、最终输出格式 (严格遵守)**

```json
{
  "resource_recommendations": [
    {
      "title": "string",
      "score": 9.8,
      "pinned_comment": "string"
    }
  ]
}
```
"""

def diff_replay_lists(current_replay_info, local_replay_info):
    """
    返回 (only_in_current, only_in_local)
    比较准则：只有当 'title','reply','key1','key2' 四个字段完全相同才视为相同条目（逐字符比较）。
    返回的列表保持原始输入顺序，元素为原始字典（未深拷贝，但未修改）。
    """
    def key(item):
        return (
            item.get('title'),
            item.get('reply'),
            item.get('key1'),
            item.get('key2'),
        )
    local_keys = set(key(i) for i in local_replay_info)
    current_keys = set(key(i) for i in current_replay_info)
    only_in_current = [i for i in current_replay_info if key(i) not in local_keys]
    only_in_local = [i for i in local_replay_info if key(i) not in current_keys]
    return only_in_current, only_in_local

def maintenance_replay(cookie):
    """
    维护指定用户的自动回复关键词，会和replay_info.json严格保持一致
    """
    try:
        result = update_bili_user_sign(cookie, "私信发送 福利 有惊喜哦！！")
        print(f"更新用户签名结果: {result}")
        # 打开自动回复
        keys_reply_value = '1'
        result = set_bili_keys_reply(keys_reply=keys_reply_value, cookie_str=cookie)
        print(f"设置自动回复状态为 {keys_reply_value} 结果: {result}")

        local_replay_info_file = f'{BASE_DIR}/replay_info.json'
        local_replay_info = read_json(local_replay_info_file)
        current_replay_info = get_bili_reply_text(cookie)
        current_replay_info = current_replay_info.get('data', {}).get('texts', [])
        only_in_current, only_in_local = diff_replay_lists(current_replay_info, local_replay_info)
        print(f"当前回复关键词数量: {len(current_replay_info)} 待删除的数量: {len(only_in_current)} 待添加的数量: {len(only_in_local)}")

        for current in only_in_current:
            target_id = current.get('id')
            result = del_bili_reply_text(target_id, cookie)
            print(f"删除回复关键词 {target_id} 结果: {result}")


        for local in only_in_local:
            title = local.get('title', '')
            reply = local.get('reply', '')
            key1 = local.get('key1', '')
            key2 = key1
            result = set_bili_reply(title=title, reply=reply, key1=key1, key2=key2, cookie_str=cookie)
            print(f'添加回复关键词 {title} 结果: {result}')

        total_title = '汇总'
        total_key1 = '惊喜，学习，材料，福利'
        total_key2 = '惊喜，学习，材料，福利'

        total_reply = ''
        for local_replay in local_replay_info:
            reply = local_replay.get('reply', '')
            total_reply += reply + '\n\n'
        result = set_bili_reply(title=total_title, reply=total_reply, key1=total_key1, key2=total_key2, cookie_str=cookie)
        print(f'添加总回复关键词 {total_title} 结果: {result}')
    except Exception as e:
        print(f"维护回复关键词时发生错误: {e}")
        traceback.print_exc()

def load_all_replay_info():
    local_replay_info_file = f'{BASE_DIR}/replay_info.json'
    local_replay_info = read_json(local_replay_info_file)
    for local_replay in local_replay_info:
        title = local_replay.get('title', '')
        detail_info_path = f"{BASE_DIR}/{title}.json"
        detail_info = read_json(detail_info_path)
        local_replay['detail_info'] = detail_info
    return local_replay_info

def gen_final_property_replay(video_info, all_replay_info):
    """
    根据视频信息生成合适的商品信息
    """
    format_property_replay_list = []
    for replay in all_replay_info:
        temp_dict = {}
        temp_dict['title'] = replay.get('title', '')
        temp_dict['detail_info'] = replay.get('detail_info', '')
        format_property_replay_list.append(temp_dict)
    print(f"正在生成最终商品信息，视频信息")
    retry_delay = 10
    max_retries = 3
    format_video_info = {}
    title_schemes = video_info.get('title_schemes', [])
    titles = format_title(title_schemes)
    format_video_info['titles'] = titles

    danmu_info = video_info.get('danmu_info') or {}
    video_anlyse = danmu_info.get('视频分析', {})
    format_video_info['video_anlyse'] = video_anlyse

    comment_list = video_info.get('hudong', {}).get('comment_list', [])
    temp_comments = [(c[0], c[1]) for c in comment_list]
    # 按照c[1]降序排序，截取前100
    temp_comments = sorted(temp_comments, key=lambda x: x[1], reverse=True)[:100]

    format_video_info['comments'] = temp_comments

    format_video_info['resources'] = format_property_replay_list

    prompt = f"{final_prompt}\n输入信息如下:\n{format_video_info}"

    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = get_llm_content(prompt=prompt, model_name="gemini-2.5-flash-lite")
            video_info = string_to_object(raw)
            return video_info
        except Exception as e:
            print(f"[ERROR] 生成视频信息失败 (尝试 {attempt}/{max_retries}): {e} {raw}")
            if attempt < max_retries:
                print(f"[INFO] 正在重试... (等待 {retry_delay} 秒)")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print("[ERROR] 达到最大重试次数，失败.")
                return None  # 达到最大重试次数后返回 None
            traceback.print_exc()


def _process_single_video(
        bvid: str,
        record: Dict[str, Any],
        commenter_pool
) -> Dict[str, Any]:
    """处理单个视频的回复逻辑，返回更新后的记录。"""
    updated_record = record.copy()

    # 1. 检查目标评论是否存在
    rpid = updated_record.get('rpid')
    if not rpid:
        print(f"视频 {bvid} 缺少 rpid，无法处理。")
        return updated_record  # 直接返回，不标记删除

    comments = get_bilibili_comments(bvid)
    # 假设 get_bilibili_comments 在失败时返回 None 或空列表
    if not _is_rpid_in_comments(rpid, comments):
        print(f"视频 {bvid} 的目标评论 {rpid} 不存在或已删除，标记为删除。")
        updated_record['status'] = 'delete'
        return updated_record

    exist_shill_comments = updated_record.get('exist_shill_comments', [])
    all_shill_comments = updated_record.get('shill_comments', [])
    comments_to_send = [c for c in all_shill_comments if c not in exist_shill_comments]

    if not comments_to_send:
        print(f"视频 {bvid} 没有新的评论文案可以发送。")
        return updated_record

    # 3. 确定本次运行最大回复数
    max_replies_this_run = (REDUCED_MAX_REPLIES_PER_RUN
                            if len(exist_shill_comments) >= EXISTING_REPLIES_THRESHOLD
                            else DEFAULT_MAX_REPLIES_PER_RUN)

    # 4. 执行回复
    success_count = 0
    exist_shill_users = updated_record.get('exist_shill_users', [])

    for shill_comment in comments_to_send:
        if success_count >= max_replies_this_run:
            print(f"已达到本次运行回复上限 ({max_replies_this_run})，停止回复视频 {bvid}。")
            break

        # 每次都随机化评论员，但排除已用过的
        available_commenters = [c for c in commenter_pool if c[0] not in exist_shill_users]
        random.shuffle(available_commenters)

        for commenter_name, commenter in available_commenters:
            reply_rpid, reason = commenter.reply_to_comment(
                bvid=bvid,
                message_content=shill_comment,
                root_rpid=rpid,
                parent_rpid=rpid
            )

            if reply_rpid:
                success_count += 1
                exist_shill_comments.append(shill_comment)
                exist_shill_users.append(commenter_name)
                print(f"✅ {commenter_name} 回复成功: 视频 {bvid}, 内容: {shill_comment[:30]}...")
                # 成功后，此条文案完成，换下一条
                break
            else:
                print(f"❌ {commenter_name} 回复失败: {reason}")
                time.sleep(RETRY_INTERVAL_SECONDS)  # 等待一段时间再重试
                if '无法获取有效的视频信息' in reason or '删除' in reason:
                    print(f"视频 {bvid} 或评论似乎已失效，标记为删除。")
                    updated_record['status'] = 'delete'
                    # 如果视频失效，直接终止对该视频的所有操作
                    return updated_record

                time.sleep(RETRY_INTERVAL_SECONDS)

    updated_record['exist_shill_comments'] = exist_shill_comments
    updated_record['exist_shill_users'] = exist_shill_users
    return updated_record


def auto_replay_refactored(user_name: str):
    """
    自动扫描并回复指定用户的置顶评论，以增加商品购买几率。
    重构版本：逻辑清晰，职责分离，性能更优。
    """
    print(f"\n🚀 开始为用户 {user_name} 的视频增加置顶文案回复...当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 加载数据与配置
    try:
        all_records_file = f"{BASE_DIR}/{user_name}_replay_record_info.json"
        all_records = read_json(all_records_file)
        config_map = init_config()
    except FileNotFoundError:
        print(f"错误：找不到文件 {all_records_file} 或配置文件。")
        return
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    # 2. 初始化评论员
    commenters = _initialize_commenters(config_map, user_to_exclude=user_name)
    if not commenters:
        print("没有可用的评论员账号，程序终止。")
        return
    commenter_pool = list(commenters.items())

    # 3. 循环处理每个视频
    today = datetime.date.today().isoformat()
    processed_count = 0
    total_records = len(all_records)

    for bvid, record in all_records.items():
        processed_count += 1
        print(f"\n[{processed_count}/{total_records}] 正在处理视频 BVID: {bvid}...")

        # 3.1. 前置检查
        skip_reason = _should_skip_video(record, bvid, today)
        if skip_reason:
            print(f"⏭️  跳过: {skip_reason}")
            # 注意：即使跳过，我们也更新处理日期和次数，防止无限次检查
            record['last_processed_date'] = today
            # 仅在非“达到上限”原因跳过时增加处理次数
            if "已达上限" not in skip_reason:
                record['process_count'] = record.get('process_count', 0) + 1
            all_records[bvid] = record
            continue

        # 3.2. 核心处理逻辑
        try:
            updated_record = _process_single_video(bvid, record, commenter_pool)
            # 无论处理结果如何，都更新处理日期和次数
            updated_record['last_processed_date'] = today
            updated_record['process_count'] = updated_record.get('process_count', 0) + 1
            all_records[bvid] = updated_record
            save_json_safe(all_records_file, all_records)

        except Exception as e:
            print(f"处理视频 {bvid} 时发生未知严重错误: {e}")
            traceback.print_exc()
            # 同样更新记录，防止下次再次出错
            record['last_processed_date'] = today
            record['process_count'] = record.get('process_count', 0) + 1
            all_records[bvid] = record
            save_json_safe(all_records_file, all_records)

    # 4. 一次性保存所有更改
    try:
        save_json_safe(all_records_file, all_records)
        print(f"\n✅ 全部处理完成，已将更新后的 {len(all_records)} 条记录保存至文件。")
    except Exception as e:
        print(f"最终保存文件时出错: {e}")

def send_replay_comment(
    commenter: Any,
    bvid: str,
    record_info
):
    """
    发送商品评论到指定的 B 站视频，并将评论置顶。

    Args:
        commenter: 带有 post_comment 和 pin_comment 方法的对象。
        bvid: 视频的 BVID 字符串。
        final_goods_record: 包含 'final_goods' 和 'property_goods' 键的字典，
                            其中 'product_recommendations' 是待推荐商品列表。
    """
    all_replay_info = record_info['property_goods']
    final_goods = record_info.get('final_goods', {})
    print(f"\n\n正在发送回复性评论到视频 {bvid}")
    recommendations = final_goods.get('resource_recommendations', [])
    sorted_recs = sorted(
        [
            item for item in recommendations
            if (float(item.get('estimated_ctr') or 1) * float(item.get('score') or 0)) >= 0
        ],
        key=lambda item: float(item.get('estimated_ctr') or 1) * float(item.get('score') or 0),
        reverse=True
    )
    print(f"找到 {len(sorted_recs)} 条商品推荐，按预估点击率和评分排序。过滤前推荐数量: {len(recommendations)}")
    # 2. 获取完整商品信息列表
    property_goods = all_replay_info

    for rec in sorted_recs:
        title: str = rec.get('title', '')
        if not title:
            continue

        # 3. 在 property_goods 找到对应商品
        target_good: Optional[Dict[str, Any]] = next(
            (pg for pg in property_goods if pg.get('title') == title),
            None
        )
        if not target_good:
            target_good = property_goods[0]
        abd_image_path = target_good.get('abd_image_path', '')

        abd_image_path_list = target_good.get('abd_image_path_list', [])
        if abd_image_path_list:
            abd_image_path = random.choice(abd_image_path_list)
        # abd_image_path = ""
        pinned_text: str = rec.get('pinned_comment', '').strip()
        key1 = target_good.get('key1', '')
        key_list = key1.split('，')
        message_list = target_good.get('message', [])
        message = random.choice(message_list) if message_list else ''
        comment_body = f"{pinned_text}\n{message}\n资料已经整理完毕，私信回复 {key_list[0]} 这{len(key_list[0])}个字，即可领取"
        # comment_body = f"{pinned_text}\n{message}"

        # 4. 发布评论
        print(f"正在发布商品评论: 视频 {bvid}，商品 {title} “{rec.get('title', '')}” comment_body: {comment_body}")
        if os.path.exists(abd_image_path):
            rpid = commenter.post_comment(bvid=bvid, message_content=comment_body, image_path=abd_image_path)
        else:
            rpid = commenter.post_comment(bvid=bvid, message_content=comment_body)
        if not rpid:
            # 发布失败，尝试下一个
            continue

        # 5. 置顶评论并结束
        if commenter.pin_comment(bvid=bvid, rpid=rpid):
            record_info['comment_body'] = comment_body
            shill_comments = target_good.get('shill_comments', [])
            # 将shill_comments打乱
            random.shuffle(shill_comments)
            record_info['shill_comments'] = shill_comments
            print(f"✅ 已成功发送并置顶商品评论: 视频 {bvid}，商品 {title} “{rec.get('title', '')}” comment_body: {comment_body}")
            time.sleep(60)
            return rpid, target_good.get('title', '')

    # 如果所有推荐都处理完仍未成功
    print(f"⚠️ 未能发送或置顶任何商品评论到视频 {bvid}")
    return None, None

def add_replay_comment_for_video(user_name='qiqi'):
    """
    为视频增加合适的商品链接
    """
    all_replay_info = load_all_replay_info()
    bvid_file_path = 'bvid_file.json'
    bvid_file_data = read_json(bvid_file_path)
    print(f"\n\n开始为用户 {user_name} 的视频增加网盘评论...")
    config_map = init_config()
    all_records_file = f"{BASE_DIR}/{user_name}_replay_record_info.json"
    # 找到对应的 UID
    uid = '1223805908'
    for key, value in config_map.items():
        if value['name'] == user_name:
            uid = key
            break
    # update_local_goods_info(user_name)
    total_cookie = config_map[uid]['total_cookie']
    csrf_token = config_map[uid].get('BILI_JCT', '')
    all_params = config_map[uid].get('all_params', {})
    commenter = BilibiliCommenter(total_cookie=total_cookie, csrf_token=csrf_token,all_params=all_params)
    temp_found_videos = bvid_file_data.get(user_name, [])
    temp_found_videos = temp_found_videos[:10]
    metadata_cache_with_uploads_back = read_json('../../LLM/TikTokDownloader/metadata_cache_with_uploads.json')
    metadata_cache_with_uploads = read_json(
        '../../LLM/TikTokDownloader/metadata_cache_with_uploads0824.json')
    metadata_cache_with_uploads.update(metadata_cache_with_uploads_back)
    all_records = read_json(all_records_file)
    success_bvids = []
    for rec in all_records.values():
        bvid = rec.get('bvid')
        if not bvid:
            continue
        if rec.get('status') == 'success' and rec.get('rpid'):
            success_bvids.append(bvid)
            continue
        try:
            if int(rec.get('process_count', 0)) > 1:
                success_bvids.append(bvid)
        except (TypeError, ValueError):
            # process_count 非整数字符时视为 0，忽略
            pass

    processed_bvids = set(success_bvids)

    print(f"已处理 {len(all_records)} 条记录，其中 {len(success_bvids)} 条成功。")
    # 过滤出已经处理过的
    videos_to_process = [video for video in temp_found_videos if video['bvid'] not in processed_bvids]
    print(f"{user_name} 找到 {len(videos_to_process)} 个未处理的视频。总共视频数量：{len(temp_found_videos)}")
    videos_to_process = videos_to_process[:10]
    for video in videos_to_process:
        try:
            bvid = video['bvid']

            target_value = find_video_by_bvid(bvid, metadata_cache_with_uploads) or {}
            if not target_value:
                print(f"视频 {bvid} 在 metadata_cache_with_uploads.json 中未找到对应信息，跳过。")
                continue
            print(f"\n\n正在处理视频 {bvid}，标题: {target_value.get('original_url', '未知标题')}")
            record = all_records.get(bvid, {})
            if bvid not in all_records:
                all_records[bvid] = {}
            all_records[bvid]['bvid'] = bvid
            all_records[bvid]['user_name'] = user_name
            save_json_safe(all_records_file, all_records)
            if 'final_goods' in record and record['final_goods'] and False:
                print(f"视频 {bvid} 已经有最终商品信息，跳过。")
                final_goods = record['final_goods']
            else:
                final_goods = gen_final_property_replay(target_value, all_replay_info)
                all_records[bvid]['final_goods'] = final_goods
                save_json_safe(all_records_file, all_records)
            if final_goods:
                all_records[bvid]['property_goods'] = all_replay_info
                rpid, title = send_replay_comment(commenter, bvid, all_records[bvid])
                if rpid:
                    all_records[bvid]['status'] = 'success'
                    all_records[bvid]['rpid'] = rpid
                    all_records[bvid]['title'] = title
                    all_records[bvid]['upload_time'] = time.time()
                    all_records[bvid]['send_time'] = time.time()
                    all_records[bvid]['property_goods'] = []
                    save_json_safe(all_records_file, all_records)
        except Exception as e:
            print(f"处理视频 {bvid} 时出错: {e}")
            traceback.print_exc()
            all_records[bvid]['status'] = 'error'
            all_records[bvid]['error_message'] = str(e)
            save_json_safe(all_records_file, all_records)

def gen_all_type_image():
    all_replay_info = load_all_replay_info()
    current_date = datetime.date.today().isoformat()
    for replay in all_replay_info:
        abd_image_path_list = []
        title = replay.get('title1', '')
        key_list = replay.get('key1', '').split('，')
        cover_path = replay.get('abd_image_path', '')
        message_list = replay.get('message', [])
        for key in key_list:
            for i in range(30):
                output_image_path = cover_path.replace('.jpg', f'_{key}_{i}.jpg')
                message = random.choice(message_list) if message_list else ''
                create_enhanced_cover(
                    input_image_path=cover_path,
                    position='center',
                    output_image_path=output_image_path,
                    text_lines=[f'私信回复 {key} 获取资料', '', title, '', message, f"{current_date}最新整理"],
                )
                abd_image_path_list.append(output_image_path)
        replay['abd_image_path_list'] = abd_image_path_list
    save_json_safe(f'{BASE_DIR}/replay_info.json', all_replay_info)


def process_user(user):
    """子进程执行逻辑"""
    try:
        start_time = time.time()
        print(f"[{time.strftime('%X')}] 子进程开始处理用户: {user}")
        add_replay_comment_for_video(user)
        # auto_replay_refactored(user)
        print(f"[{time.strftime('%X')}] 子进程完成用户: {user} 处理，耗时: {time.time() - start_time:.2f} 秒")
    except Exception as e:
        print(f"[{time.strftime('%X')}] 子进程处理用户 {user} 时出错: {e}")
        traceback.print_exc()


def run_once(username_list):
    print(f"当前配置的用户列表:{len(username_list)}个 {username_list}")

    print("--- 主进程启动，准备以 2 个并行进程处理用户 ---")
    with Pool(processes=2) as pool:
        pool.map(process_user, username_list)

    print("--- 所有用户处理完成 ---")

def test_comment():
    config = init_config()
    commenters = _initialize_commenters(config, user_to_exclude='oo')
    comment_text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
    comment_type = 1
    # for key, commenter in commenters.items():
    #     posted_rpid = commenter.post_comment(
    #         'BV1c3t1zDEzw', comment_text, comment_type,
    #         like_video=True)

    for key, commenter in commenters.items():
        print("-" * 30)
        print("步骤 3: 尝试发送一条弹幕...")
        danmaku_text = f"大家怎么样，心情都好"
        danmaku_time_ms = 1000
        danmaku_sent = commenter.send_danmaku(
            bvid='BV1c3t1zDEzw', msg=danmaku_text, progress=danmaku_time_ms, is_up=False
        )
        if danmaku_sent:
            print(f"{key} 弹幕发送流程成功完成！")
        else:
            print(f"{key} 弹幕发送流程失败。")

if __name__ == '__main__':
    # test_comment()


    # gen_all_type_image()


    username_list = [ 'mama', 'nana', 'ruru', 'jie', 'jun', 'xiaosu', 'yan', 'xiaohao', 'chabian']
    cookie_list = []
    # username_list = ['jie']
    username_list = ['mama']
    config = init_config()
    # for key,value in config.items():
    #     # if 'ruru' == value['name']:
    #     #     continue
    #     username_list.append(value['name'])
    #     cookie_list.append(value['total_cookie'])

    #
    # for cookie in cookie_list:
    #     maintenance_replay(cookie)



    # 无限循环：每轮执行一次 run_once
    while True:
        # 打乱username_list顺序
        random.shuffle(username_list)
        start_time = time.time()
        run_once(username_list)
        # 计算本轮执行时间
        elapsed_time = time.time() - start_time
        # 至少30分钟执行
        wait_time = max(1800 - elapsed_time, 0)
        print(f"本轮执行时间: {elapsed_time:.2f} 秒，等待 {wait_time:.2f} 秒后开始下一轮...")
        time.sleep(wait_time)