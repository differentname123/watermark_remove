import json

import requests
import time
import random
import re  # 用于解析Cookie
import logging
from common_utils.common_utils import get_config, read_json, save_json
from content_community.bilibili.BiliVideoCommenter import load_processed_set

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
need_clear = True

total_cookie = get_config("ruru_bilibili_total_cookie")
FULL_COOKIE_STRING = total_cookie

# 用户代理，模拟浏览器行为
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# 每次操作（关注或取消关注）之间的延迟范围（秒）
MIN_OPERATION_DELAY_SEC = 20
MAX_OPERATION_DELAY_SEC = 45

# 获取列表时每页的数量 (最大50)
PAGE_SIZE = 50

# --- API 地址常量 ---
URL_MY_INFO = "https://api.bilibili.com/x/space/myinfo"
URL_GET_FOLLOWERS = "https://api.bilibili.com/x/relation/followers"
URL_GET_FOLLOWINGS = "https://api.bilibili.com/x/relation/followings"  # 获取我关注的人
URL_MODIFY_RELATION = "https://api.bilibili.com/x/relation/modify"

# --- 全局变量和初始化 ---
session = requests.Session()
cookies = {}
csrf_token = ""
my_uid = None


def parse_cookies_string(cookie_string):
    """从完整的Cookie字符串中解析出字典和bili_jct"""
    parsed_cookies = {}
    bili_jct_val = ""
    for pair in cookie_string.split(';'):
        if '=' in pair:
            key, value = pair.strip().split('=', 1)
            parsed_cookies[key] = value
            if key == 'bili_jct':
                bili_jct_val = value
    return parsed_cookies, bili_jct_val


def init_session():
    """初始化requests session和cookies"""
    global cookies, csrf_token
    cookies, csrf_token = parse_cookies_string(FULL_COOKIE_STRING)

    if not cookies.get('SESSDATA') or not csrf_token:
        logging.error("错误：SESSDATA 或 bili_jct (CSRF token) 未在提供的 Cookie 字符串中找到。请检查！")
        exit()

    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://space.bilibili.com/",
        "Origin": "https://space.bilibili.com",
    })
    session.cookies.update(cookies)
    logging.info("Session 初始化完成。SESSDATA 和 CSRF token 已加载。")


def get_my_uid():
    """获取当前登录用户的UID"""
    global my_uid
    logging.info("尝试获取您的 Bilibili UID...")
    try:
        response = session.get(URL_MY_INFO)
        response.raise_for_status()
        data = response.json()
        if data['code'] == 0:
            my_uid = data['data']['mid']
            logging.info(f"成功获取到您的 UID: {my_uid}")
            return my_uid
        else:
            logging.error(f"获取 UID 失败：{data['message']} (Code: {data['code']})")
    except requests.exceptions.RequestException as e:
        logging.error(f"请求获取 UID 失败: {e}")
    except ValueError:
        logging.error("UID 响应内容不是有效的 JSON。")
    return None


def get_user_list(url, vmid, page_size, list_type="用户"):
    """
    获取用户的粉丝或关注列表。
    url: API地址
    vmid: 用户的UID
    page_size: 每页数量
    list_type: 用于打印日志的类型 ('粉丝' 或 '关注')
    """
    user_mids = set()
    pn = 1
    total_count = 0
    logging.info(f"正在获取您的{list_type}列表...")

    while True:
        params = {
            "vmid": vmid,
            "pn": pn,
            "ps": page_size
        }
        try:
            response = session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data['code'] == 0:
                list_data = data['data']['list']
                total_count = data['data']['total']

                if not list_data:
                    break  # 没有更多数据了

                for user in list_data:
                    user_mids.add(user['mid'])

                logging.info(f"  已获取第 {pn} 页{list_type}数据, 当前已获取 {len(user_mids)}/{total_count} 条。")

                if len(user_mids) >= total_count:
                    break

                pn += 1
                time.sleep(random.uniform(0.5, 1.5))  # 小延迟
            else:
                logging.error(f"获取{list_type}列表失败 (页码 {pn})：{data['message']} (Code: {data['code']})")
                break
        except requests.exceptions.RequestException as e:
            logging.error(f"请求{list_type}列表失败 (页码 {pn}): {e}")
            break
        except ValueError:
            logging.error(f"{list_type}列表响应内容不是有效的 JSON (页码 {pn})。")
            break

    logging.info(f"成功获取到 {len(user_mids)} 个{list_type}。")
    return user_mids, total_count


def modify_relation(fid, action_type):
    """
    修改用户关系 (关注或取消关注)。
    fid: 目标用户的UID
    action_type: 1 为关注, 2 为取消关注
    """
    if action_type == 1:
        action_text = "关注"
    elif action_type == 5:
        action_text = "拉黑"
    else:
        action_text = "取消关注"

    payload = {
        "fid": fid,
        "act": action_type,
        "re_src": 11,  # 关系来源，通常用 11
        "csrf": csrf_token
    }
    try:
        response = session.post(URL_MODIFY_RELATION, data=payload)
        response.raise_for_status()
        result = response.json()
        if result['code'] == 0:
            logging.info(f"  {'✅' if action_type == 1 else '🗑️'} 成功{action_text} UID: {fid}")
            return True
        else:
            logging.error(f"  ❌ {action_text} UID: {fid} 失败: {result['message']} (Code: {result['code']})")
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f"  ❌ 请求{action_text} UID: {fid} 失败: {e}")
        return False
    except ValueError:
        logging.error(f"  ❌ {action_text} UID: {fid} 响应内容不是有效的 JSON。")
        return False


def load_followers_set(filename):
    """尝试从指定的 JSON 文件中加载之前的粉丝列表，返回 set"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            logging.info(f"已加载 {filename} 中的 {len(data)} 条记录。")
            return set(data)
    except FileNotFoundError:
        logging.info(f"{filename} 不存在，返回空集合。")
        return set()
    except json.JSONDecodeError as e:
        logging.error(f"加载 {filename} 时发生 JSON 解析错误: {e}")
        return set()


def save_followers_set(filename, followers_set):
    """保存粉丝列表到指定的 JSON 文件中（覆盖写入）"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(list(followers_set), f, ensure_ascii=False, indent=4)
        logging.info(f"粉丝列表已成功保存到 {filename}")
    except Exception as e:
        logging.error(f"保存粉丝列表到 {filename} 失败: {e}")


def update_followers(new_followers_iterable, path: str = "followers_fids.json"):
    """
    new_followers_iterable: 新抓取到的 followers（可以是 set/list/iterable）
    结果会以 JSON list 的形式保存到 path，且新出现的 ID 会追加到文件末尾（并去重）。
    """
    # 读已有（保持原有顺序）
    previous_list = read_json(path)
    previous_set = set(previous_list)

    # 保证 new_followers 有序（保持传入顺序），并过滤掉已有的
    new_followers_list = list(new_followers_iterable)
    new_added = [fid for fid in new_followers_list if fid not in previous_set]

    if new_added:
        logging.info(f"新添加的粉丝UID示例（最多5个）: {new_added[:5]}")

    # 把新加的追加到旧列表末尾
    updated_list = list(previous_list) + new_added

    # 去重
    updated_list = list(dict.fromkeys(updated_list))

    logging.info(
        f"之前的粉丝记录: {len(previous_list)} 条，" 
        f"新抓取: {len(new_followers_list)} 条，" 
        f"合并后: {len(updated_list)} 条。"
    )

    save_json(path, updated_list)


def main_task():
    """
    主任务：先清理非互关用户，再回关新粉丝
    """
    # 初始化 session 和登录信息
    init_session()
    uid = get_my_uid()
    if not uid:
        logging.error("无法获取您的 UID，本次任务终止。")
        return

    # --- 阶段 1: 清理非互关用户 ---
    logging.info("\n--- 阶段 1: 开始清理非互关用户 ---")
    followers_set, followers_total_count = get_user_list(URL_GET_FOLLOWERS, uid, PAGE_SIZE, "粉丝")
    if not need_clear:
        update_followers(followers_set)
    followings_set = set()
    if need_clear:
        followings_set, followings_total_count = get_user_list(URL_GET_FOLLOWINGS, uid, PAGE_SIZE, "关注")
    followings_set = {fid for fid in followings_set}

    followers_fids_set = load_processed_set("followers_fids.json")
    processed_fids_set = load_processed_set("target_processed_fids.json")

    non_mutual_followings = followings_set - followers_set - followers_fids_set
    if need_clear:
        non_mutual_followings = followings_set - followers_set
        more_count = followers_total_count - 1000
        need_add_count = max(0, more_count - len(non_mutual_followings))
        print(f"当前粉丝数: {followers_total_count}, 需要额外添加的非互关用户数: {need_add_count}")
        if need_add_count > 0:
            # 计算最大能添加的数量，必须保证剩下至少 50 个
            max_add_count = max(0, len(followings_set) - 50)

            # 实际需要添加的数量不能超过 max_add_count
            safe_add_count = min(need_add_count, max_add_count)

            additional_to_add = set(list(followings_set)[:safe_add_count])

            non_mutual_followings.update(additional_to_add)
            logging.info(f"为了达到清理目标，额外添加了 {len(additional_to_add)} 位非互关用户进行清理。")


    if not non_mutual_followings:
        logging.info("您当前关注的人都已关注您，阶段 1 无需清理。")
    else:
        if len(followings_set) > 4000 or need_clear:
            logging.info(f"--- 发现 {len(non_mutual_followings)} 位您已关注但未回关的用户 ---")
            non_mutual_followings_list = list(non_mutual_followings)
            random.shuffle(non_mutual_followings_list)
            logging.info("自动开始取消关注操作。")

            successful_unfollows = 0
            failed_unfollows = 0
            for i, fid in enumerate(non_mutual_followings_list):
                logging.info(f"\n正在取消关注第 {i + 1}/{len(non_mutual_followings_list)} 位用户 (UID: {fid})...")
                if modify_relation(fid, 2):  # 2 代表取消关注
                    successful_unfollows += 1
                else:
                    failed_unfollows += 1
                # if successful_unfollows > 600:
                #     logging.info("已取消关注超过 500 人，停止后续操作。")
                #     break

                delay = random.uniform(MIN_OPERATION_DELAY_SEC, MAX_OPERATION_DELAY_SEC)
                delay = delay / 10
                logging.info(f"等待 {delay:.2f} 秒...")
                time.sleep(delay)

            logging.info("\n--- 阶段 1: 清理操作完成 ---")
            logging.info(f"总计尝试取消关注: {len(non_mutual_followings_list)} 人")
            logging.info(f"成功取消关注: {successful_unfollows} 人, 失败: {failed_unfollows} 人")
    if need_clear:
        return
    # --- 阶段 2: 回关粉丝 ---
    logging.info("\n--- 阶段 2: 开始回关新粉丝 ---")
    # 重新获取最新的列表，因为阶段1可能已经更改了关注状态
    new_followers_set, total_count = get_user_list(URL_GET_FOLLOWERS, uid, PAGE_SIZE, "粉丝")
    if not need_clear:
        update_followers(new_followers_set)
    new_followings_set, total_count = get_user_list(URL_GET_FOLLOWINGS, uid, PAGE_SIZE, "关注")
    # 将new_followings_set的元素全部变成字符串形式

    # 1. 加载两个来源的数据到独立的集合
    followers_fids_set = read_json("followers_fids.json")
    # 将followers_fids_set逆序
    followers_fids_set.reverse()
    processed_fids_set = load_processed_set("target_processed_fids.json")

    # 2. 创建一个空列表，用于存放最终需要 follow 的 FID (保持顺序)
    followers_to_follow_list = []
    failed_set = load_processed_set("failed_set.json")

    # 3. 使用一个集合来跟踪已经添加到列表中的 FID，避免重复
    already_added_to_list = []

    # 4. **优先处理来自 followers_fids.json 的 FIDs**
    #    将那些不在 new_followings_set 中的添加到列表中
    print("\n--- Identifying Prioritized FIDs to follow ---")
    for fid in followers_fids_set:
        if fid not in new_followings_set:
            if fid not in already_added_to_list and fid not in failed_set:  # 确保不会重复添加 (虽然这里不应该重复)
                followers_to_follow_list.append(fid)
                already_added_to_list.append(fid)
                # print(f"Added prioritized FID: {fid}") # 可选：用于调试

    # # 5. **处理来自 processed_fids.json 的剩余 FIDs**
    # #    将那些不在 new_followings_set 中，并且还没有被添加到列表中的 FID 添加
    # print("--- Identifying Remaining FIDs to follow ---")
    # for fid in processed_fids_set:
    #     if fid not in new_followings_set:
    #         if fid not in already_added_to_list and fid not in failed_set:  # 确保不会重复添加 (虽然这里不应该重复)
    #             followers_to_follow_list.append(fid)
    #             already_added_to_list.add(fid)
    followers_to_follow = already_added_to_list
    if not followers_to_follow:
        logging.info("所有粉丝均已关注，阶段 2 无需操作。")
    else:
        logging.info(f"--- 发现 {len(followers_to_follow)} 位未回关的粉丝 ---")
        followers_to_follow_list = list(followers_to_follow)
        random.shuffle(followers_to_follow_list)
        logging.info("自动开始回关操作。")

        successful_follows = 0
        failed_follows = 0
        for i, fid in enumerate(followers_to_follow_list):
            logging.info(f"\n正在回关第 {i + 1}/{len(followers_to_follow_list)} 位粉丝 (UID: {fid})...")
            if modify_relation(fid, 1):  # 1 代表关注
                successful_follows += 1
            else:
                time.sleep(1200)
                failed_follows += 1
                failed_set.add(fid)  # 将失败的 FID 添加到集合中
            if successful_follows > 5000:
                logging.info("已回关超过 500 人，停止后续操作。")
                break

            delay = random.uniform(MIN_OPERATION_DELAY_SEC, MAX_OPERATION_DELAY_SEC)
            delay = delay / 2
            logging.info(f"等待 {delay:.2f} 秒...")
            time.sleep(delay)
        # 删除followers_fids_set和processed_fids_set中失败的 FID
        save_followers_set("failed_set.json", failed_set)

        logging.info("\n--- 阶段 2: 回关操作完成 ---")
        logging.info(f"总计尝试回关: {len(followers_to_follow_list)} 人")
        logging.info(f"成功回关: {successful_follows} 人, 失败: {failed_follows} 人")

    logging.info("\n所有操作执行完毕。")


if __name__ == "__main__":
    # 使用 robust 的定时任务调度，每个小时执行一次
    while True:
        start_time = time.time()
        logging.info("定时任务开始执行...")
        try:
            main_task()
        except Exception as e:
            logging.exception("任务执行过程中发生异常：")
        elapsed = time.time() - start_time
        sleep_duration = max(0, 3600)
        logging.info(f"任务执行完毕，耗时 {elapsed:.2f} 秒，等待 {sleep_duration:.2f} 秒后开始下一次执行。")
        time.sleep(sleep_duration)