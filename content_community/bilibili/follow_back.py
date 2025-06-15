import requests
import time
import random
import re  # 用于解析Cookie
import logging
from common_utils.common_utils import get_config
from content_community.bilibili.BiliVideoCommenter import load_processed_set

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

total_cookie = get_config("nana_bilibili_total_cookie")
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
    return user_mids

def modify_relation(fid, action_type):
    """
    修改用户关系 (关注或取消关注)。
    fid: 目标用户的UID
    action_type: 1 为关注, 2 为取消关注
    """
    action_text = "关注" if action_type == 1 else "取消关注"
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

    # # --- 阶段 1: 清理非互关用户 ---
    # logging.info("\n--- 阶段 1: 开始清理非互关用户 ---")
    # followers_set = get_user_list(URL_GET_FOLLOWERS, uid, PAGE_SIZE, "粉丝")
    # followings_set = get_user_list(URL_GET_FOLLOWINGS, uid, PAGE_SIZE, "关注")
    # non_mutual_followings = followings_set - followers_set
    #
    # if not non_mutual_followings:
    #     logging.info("您当前关注的人都已关注您，阶段 1 无需清理。")
    # else:
    #     logging.info(f"--- 发现 {len(non_mutual_followings)} 位您已关注但未回关的用户 ---")
    #     non_mutual_followings_list = list(non_mutual_followings)
    #     random.shuffle(non_mutual_followings_list)
    #     logging.info("自动开始取消关注操作。")
    #
    #     successful_unfollows = 0
    #     failed_unfollows = 0
    #     for i, fid in enumerate(non_mutual_followings_list):
    #         logging.info(f"\n正在取消关注第 {i + 1}/{len(non_mutual_followings_list)} 位用户 (UID: {fid})...")
    #         if modify_relation(fid, 2):  # 2 代表取消关注
    #             successful_unfollows += 1
    #         else:
    #             failed_unfollows += 1
    #
    #         delay = random.uniform(MIN_OPERATION_DELAY_SEC, MAX_OPERATION_DELAY_SEC)
    #         logging.info(f"等待 {delay:.2f} 秒...")
    #         time.sleep(delay)
    #
    #     logging.info("\n--- 阶段 1: 清理操作完成 ---")
    #     logging.info(f"总计尝试取消关注: {len(non_mutual_followings_list)} 人")
    #     logging.info(f"成功取消关注: {successful_unfollows} 人, 失败: {failed_unfollows} 人")

    # --- 阶段 2: 回关粉丝 ---
    logging.info("\n--- 阶段 2: 开始回关新粉丝 ---")
    # 重新获取最新的列表，因为阶段1可能已经更改了关注状态
    new_followers_set = get_user_list(URL_GET_FOLLOWERS, uid, PAGE_SIZE, "粉丝")
    new_followings_set = get_user_list(URL_GET_FOLLOWINGS, uid, PAGE_SIZE, "关注")
    new_followers_set.update(load_processed_set("processed_fids.json"))
    followers_to_follow = new_followers_set - new_followings_set

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
                failed_follows += 1

            delay = random.uniform(MIN_OPERATION_DELAY_SEC, MAX_OPERATION_DELAY_SEC)
            logging.info(f"等待 {delay:.2f} 秒...")
            time.sleep(delay)

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