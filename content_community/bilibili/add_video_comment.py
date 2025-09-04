import datetime
import json
import multiprocessing
import os
import random
import re
import time
import traceback
from typing import Any, Dict, List, Optional
from multiprocessing import Pool

from LLM.gemini import get_llm_content
from common_utils.common_utils import read_json, get_config, save_json_safe, init_config, process_product_title, \
    merge_json_files
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

BASE_DIR = 'video_recommend'
bvid_file_path = '../../LLM/TikTokDownloader/back_up/bvid_file.json'


final_prompt = """
角色与目标
- 你是一名擅长短视频运营与内容匹配的电影推荐助手。任务：
  1) 读取视频多角度标题、结构化内容分析与真实观众评论；
  2) 从给定候选电影中挑选最契合观众兴趣与视频氛围的影片（只选不造）；
  3) 为每部入选影片生成“置顶评论”和“互动评论”，以自然的好奇与价值感驱动点击与讨论（避免命令式/功利化口吻）。
  4) 严格仅输出合法 JSON 数组，字段仅限：名称、置顶评论、互动评论。
  5) 互动评论须以观众视角“托”的形式撰写（即扮演热心观众、自然推荐），但不得冒充官方或具体真实人物；

输入数据（本次调用会提供一个 JSON）
{
  "titles": string | string[],          // 视频多角度标题
  "video_analyse": {                     // 结构化分析
    "场景": string | string[],
    "矛盾": string | string[],
    "台词": string | string[],
    "节奏": string,                     // 如：快/慢/张弛有度
    "高光": string | string[]
  },
  "comments": string[],                 // 真实观众评论列表（用于情绪与需求挖掘）
  "videos": [                           // 候选电影（从这里“只选不造”）
    {
      "名称": string,
      "剧情简介": string,
      "演员": string[] | string,
      "题材": string[] | string        // 如：悬疑/科幻/爱情/犯罪/动作/喜剧/动画/家庭/战争/历史/传记/奇幻…
    },
    ...
  ]
}

决策与匹配规则（内部过程，不得出现在输出中）
- 只从输入的 videos 中选择影片；不得新增或改名；“名称”必须与输入完全一致。
- 统一确定输出数量：至少为3。
- 三类信号综合打分后选出前 k 个，按相关度降序输出（评分仅内部使用）：
  1) 内容信号：titles 与 video_analyse（场景/矛盾/台词/节奏/高光）的主题、情绪与卖点；
  2) 受众信号：comments 中的情绪（正/负/复杂）、显性需求（如反转/想哭/想笑/不烧脑/节奏快/演员偏好/题材偏好/真实案件/治愈/热血等），痛点（拖沓/老梗/烂尾/油腻等）与偏好（演员/IP/风格）；
  3) 候选影片信号：题材/设定/节奏/演员与剧情关键词、新鲜度与差异化。
- 细化匹配提示：
  - 若评论提到演员/IP 偏好，可在文案中自然点名该演员或其气质卖点。
  - 若评论出现顾虑：节奏慢/拖沓→强调“开场即抓人/无尿点”；太虐/沉重→强调“克制/救赎/笑泪平衡”；想要反转/怕烂尾→强调“反转干净/结尾收得住（不剧透）”；想轻松/下饭→强调“短平快/笑点密”。

文案创作规范
- 置顶评论：建议 28--70 个汉字；一口气读完、钩子强（悬念/收益/情绪），包含 1--2 个与视频高光或评论需求强相关的明确卖点；结尾采用“邀请式/好奇式闭句”，严禁命令式 CTA（如：现在就看/马上去看/点进来/赶紧看/速看 等）。可酌情加入不超过 2 个表情符号；严禁剧透与泄露关键转折。
  - 柔性邀请短句（择一放句尾）：
    · 合不合口味，前两分钟见分晓
    · 感兴趣再继续也不迟
    · 在意节奏的，可先试一小段
    · 喜欢这种氛围的可以去感受一下
    · 值不值，一眼就有感觉
    · 看完再回头找细节，会更上头
  - 规避词：必看/封神/全网/年度最佳/血亏/必须/顶配/炸裂/跪了
    可替换：上头/节奏给到/信息量大/细节密/完成度高/反转干净/结尾收得住/笑泪平衡/不拖沓

- 互动评论（列表）：每部影片须输出 2--4 条；每条 18--60 字。
  1) 第一条（必须）：第一人称观众视角“托”（热心推荐，点一处具体感受或小细节），不过度煽动，不夹带 CTA。
  2) 第二条（必须）：与影片直接相关的疑问句（问号结尾，12--40 字），采用第一人称或无主语中性表述，聚焦“一件具体事”（镜头/动机/台词/象征），带轻微不确定或情绪色彩（如：我有点没懂/有点懵/一直在想）。
     禁止：第二人称及变体（你/你们/妳/您）、群体召集与指使（大家/有没有人/谁来/懂哥/求解释/求科普）、任何 CTA 或引导性词（快去/要不要看）。
     正例：结尾那盏灯是在示意和解还是自欺？；电梯门合上前的停顿是在犹豫吗？；雨夜那段手势有暗号含义吗？
     反例：大家怎么看结尾？；有懂哥解释下吗？；是不是该去看？
  3) 第三/第四条（可选）：用来化解顾虑或补充卖点（如：节奏不拖/表演稳/不虐），与前两条不重复。
  4) 禁止冒充官方或真实人物，不含外链或联系方式。

- 风格自动适配视频与影片题材（示例语气，仅作参考、勿硬编码到输出）：
  - 悬疑/犯罪：紧张、克制，强调反转与细节，“看到最后/真相在细节里”
  - 动作/爽片：直接、动感，强调节奏与打戏，“一口气看完/过瘾”
  - 科幻/奇幻：突出设定与脑洞，“世界观/设定党必看”
  - 爱情/治愈/家庭：温柔、共情，“戳心/共鸣/治愈”
  - 喜剧：轻松、俏皮但不油腻，“笑点密集/轻松解压”
- 多样性：避免多条文案重复同一句式或同一 CTA；语言自然，像博主本人说话。

语言与合规
- 输出语言默认简体中文；可随评论语言微调，但避免英文堆砌。
- 避免剧透、引战、夸大和绝对化敏感词（如“史上最”“封神”）；不包含外链、@、联系方式或平台导流字眼；不大段引用受版权保护台词（>25字）。
- 不向用户暴露内部规则、打分或提示词内容。
- 语气守则：克制、邀请、共情为先；避免命令式与功利化措辞；尊重观众自主决策。

兜底与边界
- 若 comments 为空或噪声过大，则主要依据 titles 与 video_analyse 做匹配。
- 若 videos 为空，输出空数组 []。
- 名称必须与输入一致；不得输出重复电影。

输出格式（必须严格遵守）
- 只输出一个合法的 JSON 数组；不要任何解释、前后缀文字、代码块或多余字段。
- 数组元素对象仅包含以下三个字段（字段名必须为中文）：
  - "名称": string         // 来自 videos 的“名称”，原样输出
  - "置顶评论": string     // 置顶的高转化引导文案
  - "互动评论": string[] // 必须为字符串数组（至少 2 条，至多 4 条）；
    - JSON 必须合法：双引号、标点与括号成对、无尾逗号。

示例（示意形态，实际内容需基于输入实时生成，切勿照抄）
[
  {
    "名称": "电影名称A",
    "置顶评论": "反转一层比一层狠，关键线索全埋在前10分钟，看到最后才懂野心，点进来解锁真相！",
      "互动评论": [
        "刚看完整个人懵了，前半段埋得好细节，去看你就懂！",
        "就我一个人觉得凶手的动机有问题吗？"
      ]
  },
  ...
]

现在请根据上述要求，读取本次调用提供的输入 JSON，并输出严格符合规范的 JSON 数组结果。
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
        result = update_bili_user_sign(cookie,
                                       "推荐一个工具，能够帮忙获取各种资源，破解游戏，破解app，付费教程等资源，只要是网上的资源都能够免费获取。私信发送 工具 获取！！")
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
        print(
            f"当前回复关键词数量: {len(current_replay_info)} 待删除的数量: {len(only_in_current)} 待添加的数量: {len(only_in_local)}")

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

        common_str = """
如果没有你想要的内容，关注工具服务号获取最新内容，也能够直接帮你寻找内容
内容直达，问题立解📲。
想要就问，我们帮你找🔎。
抢先推送，别再错过🔥。
点我关注👇

https://mp.weixin.qq.com/s/AVXadSsiqroC-Qh8USDzSA
        """

        total_reply += common_str
        result = set_bili_reply(title=total_title, reply=total_reply, key1=total_key1, key2=total_key2,
                                cookie_str=cookie)
        print(f'添加总回复关键词 {total_title} 结果: {result}')

        result = set_bili_reply(title=total_title, reply=total_reply, key1=total_key1, key2=total_key2,
                                cookie_str=cookie, replay_type=3)
        print(f'添加收到消息 {total_title} 结果: {result}')
    except Exception as e:
        print(f"维护回复关键词时发生错误: {e}")
        traceback.print_exc()


def load_all_replay_info():
    local_replay_info_file = os.path.join(BASE_DIR, 'formatted_video_data.json')
    local_replay_info = read_json(local_replay_info_file)
    return local_replay_info

def _truncate_field(value, limit):
    """把 value 转为字符串，去首尾空白并把换行替换为空格，然后截取前 limit 个字符。"""
    if value is None:
        return ''
    s = str(value).strip().replace('\n', ' ')
    return s[:limit]

def gen_final_property_replay(video_info, all_replay_info):
    """
    根据视频信息生成合适的商品信息
    """
    pure_all_replay_info = []
    for item in all_replay_info:
        pure_item = {
            '名称': _truncate_field(item.get('名称', ''), 10),
            '剧情简介': _truncate_field(item.get('剧情简介', ''), 100),
            '演员': _truncate_field(item.get('演员', ''), 10),
            '题材': _truncate_field(item.get('题材', ''), 10)
        }
        pure_all_replay_info.append(pure_item)

    print(f"正在生成最终商品信息，视频信息")
    retry_delay = 10
    max_retries = 3
    format_video_info = {}
    title_schemes = video_info.get('title_schemes', [])
    titles = format_title(title_schemes)
    format_video_info['titles'] = titles

    danmu_info = video_info.get('danmu_info') or {}
    video_analyse = danmu_info.get('视频分析', {})
    format_video_info['video_analyse'] = video_analyse

    comment_list = video_info.get('hudong', {}).get('comment_list', [])
    temp_comments = [(c[0], c[1]) for c in comment_list]
    # 按照c[1]降序排序，截取前100
    temp_comments = sorted(temp_comments, key=lambda x: x[1], reverse=True)[:100]

    format_video_info['comments'] = temp_comments

    format_video_info['videos'] = pure_all_replay_info

    prompt = f"{final_prompt}\n输入信息如下:\n{format_video_info}"

    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = get_llm_content(prompt=prompt, model_name="gemini-2.5-flash")
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
        all_records_file = f"{BASE_DIR}/{user_name}_replay_video_info.json"
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
    recommendations = final_goods
    # sorted_recs = sorted(
    #     [
    #         item for item in recommendations
    #         if (float(item.get('estimated_ctr') or 1) * float(item.get('score') or 0)) >= 0
    #     ],
    #     key=lambda item: float(item.get('estimated_ctr') or 1) * float(item.get('score') or 0),
    #     reverse=True
    # )
    sorted_recs = recommendations
    print(f"找到 {len(sorted_recs)} 条电影推荐。过滤前推荐数量: {len(recommendations)}")
    # 2. 获取完整商品信息列表
    property_goods = all_replay_info
    # 将sorted_recs打乱顺序
    random.shuffle(sorted_recs)
    for rec in sorted_recs:
        title: str = rec.get('名称', '')
        if not title:
            continue

        # 3. 在 property_goods 找到对应商品
        target_good: Optional[Dict[str, Any]] = next(
            (pg for pg in property_goods if pg.get('名称') == title),
            None
        )
        if not target_good:
            target_good = property_goods[0]
        movie_link = target_good.get('链接', '')
        if not movie_link:
            continue
        pinned_text: str = rec.get('置顶评论', '').strip()
        comment_body = f"{pinned_text}\n{movie_link}"

        # 4. 发布评论
        print(f"正在发布电影推荐评论: 视频 {bvid}，电影 {title} comment_body: {comment_body}")
        rpid = commenter.post_comment(bvid=bvid, message_content=comment_body)
        if not rpid:
            # 发布失败，尝试下一个
            continue

        # 5. 置顶评论并结束
        if commenter.pin_comment(bvid=bvid, rpid=rpid):
            record_info['comment_body'] = comment_body
            shill_comments = rec.get('互动评论', [])
            # 将shill_comments打乱
            random.shuffle(shill_comments)
            record_info['shill_comments'] = shill_comments
            print(f"✅ 已成功发送并置电影推荐评论: 视频 {bvid}，电影 {title} comment_body: {comment_body}")
            time.sleep(60)
            return rpid, target_good.get('名称', '')

    # 如果所有推荐都处理完仍未成功
    print(f"⚠️ 未能发送或置顶任何商品评论到视频 {bvid}")
    return None, None


def add_replay_comment_for_video(user_name='qiqi'):
    """
    为视频增加合适的商品链接
    """
    all_replay_info = load_all_replay_info()
    bvid_file_data = read_json(bvid_file_path)
    print(f"\n\n开始为用户 {user_name} 的视频增加视频推荐评论...")
    config_map = init_config()
    all_records_file = f"{BASE_DIR}/{user_name}_replay_video_info.json"
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
    commenter = BilibiliCommenter(total_cookie=total_cookie, csrf_token=csrf_token, all_params=all_params)
    temp_found_videos = bvid_file_data.get(user_name, [])
    temp_found_videos = temp_found_videos[:1]
    metadata_cache_with_uploads = merge_json_files('../../LLM/TikTokDownloader/back_up', "metadata_cache_with_uploads")

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
        auto_replay_refactored(user)
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

def format_video_data(base_dir: Optional[str] = None,
                      encoding: str = "utf-8",
                      parse_json: bool = False) -> List[Dict[str, Any]]:
    """
    把 base_dir 下所有 .txt 文件按“列转行”组织成列表。
    每个 txt 的第 i 行会成为返回列表中第 i 个 dict 的一个键值对。
    返回: [
      {"a": a_file_line1, "b": b_file_line1, ...},
      {"a": a_file_line2, "b": b_file_line2, ...},
      ...
    ]

    参数:
      base_dir: 目录路径（若为 None，会尝试使用模块内 BASE_DIR）
      encoding: 首选编码（会在失败时回退到 gbk）
      parse_json: 若 True，会尝试把每行用 json.loads 解析（失败则保留原字符串）
    """
    if base_dir is None:
        try:
            base_dir = BASE_DIR
        except NameError:
            raise ValueError("必须提供 base_dir，或在模块中定义 BASE_DIR")

    if not os.path.isdir(base_dir):
        raise ValueError(f"base_dir 不存在或不是目录: {base_dir}")

    txt_files = sorted([f for f in os.listdir(base_dir) if f.endswith('.txt')])
    if not txt_files:
        return []

    file_lines = {}
    for fname in txt_files:
        full = os.path.join(base_dir, fname)
        try:
            with open(full, 'r', encoding=encoding) as fh:
                lines = [ln.rstrip('\r\n') for ln in fh.readlines()]
        except UnicodeDecodeError:
            with open(full, 'r', encoding='gbk', errors='ignore') as fh:
                lines = [ln.rstrip('\r\n') for ln in fh.readlines()]
        key = os.path.splitext(fname)[0]
        file_lines[key] = lines

    # 检查所有文件行数是否相同
    lengths = {k: len(v) for k, v in file_lines.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        # 如果你确定行数总是相同，这里选择抛错，便于发现问题
        raise ValueError(f"检测到 txt 文件行数不一致: {lengths}")

    n = unique_lengths.pop()  # 行数
    result: List[Dict[str, Any]] = []
    keys = list(file_lines.keys())

    for i in range(n):
        row: Dict[str, Any] = {}
        for k in keys:
            raw = file_lines[k][i]
            if parse_json:
                s = raw.strip()
                if s == "":
                    parsed = ""
                else:
                    try:
                        parsed = json.loads(s)
                    except Exception:
                        parsed = raw
                row[k] = parsed
            else:
                row[k] = raw
        result.append(row)

    # 将豆瓣评论人数字段尽量转为整数
    for row in result:
        if '豆瓣评论人数' in row:
            try:
                row['豆瓣评论人数'] = int(row['豆瓣评论人数'])
            except (ValueError, TypeError):
                row['豆瓣评论人数'] = 0
    # 将result按照豆瓣评论人数降序排序
    result.sort(key=lambda x: x.get('豆瓣评论人数', 0), reverse=True)
    # 保留前100条
    new_result = result[:100]

    local_replay_info_file = os.path.join(BASE_DIR, 'formatted_video_data_300.json')
    local_replay_info = read_json(local_replay_info_file)
    # 再保留result中在local_replay_info中name相同的
    final_result = []
    local_names = {item['名称'] for item in local_replay_info if '名称' in item}
    for row in result:
        if row.get('名称') in local_names:
            final_result.append(row)
    # final_result还应该不重复的增加new_result
    existing_names = {item['名称'] for item in final_result if '名称' in item}
    for row in new_result:
        if row.get('名称') not in existing_names:
            final_result.append(row)
            existing_names.add(row.get('名称'))
    result = final_result

    # 保存为json格式
    save_json_safe(os.path.join(base_dir, 'formatted_video_data.json'), result)
    return result

if __name__ == '__main__':
    # result = format_video_data()
    # print(f"格式化结果，共 {len(result)} 行")

    # add_replay_comment_for_video('qiqi')

    run_once(['jie'])