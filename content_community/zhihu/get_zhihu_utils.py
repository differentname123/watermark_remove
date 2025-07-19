import json
import random
import time

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import requests
from typing import List, Dict

from common_utils.common_utils import save_json

# --- 配置区域 ---
AUTH_FILE = "zhihu_auth_state.json"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.5790.110 Safari/537.36"
)
ZHIHU_COOKIE_STRING = "_xsrf=EGbVA6NHTaM3dXlCMEiWj9aRBvWl4inW; _zap=34c3bc6c-ebae-4e0d-9a06-5bfb7b8a9548; d_c0=APARO_5-wBmPTvibmi_6NacNH42miN-ERZY=|1735194921; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1752912686; HMACCOUNT=14EFD85132347319; DATE=1752912688356; crystal=U2FsdGVkX19HZsSWXHhQvCmTy0IhrzSuQUWjAmBNS9sgZFRn8/yuh3qnHWWPtqhs6Fx+lpP4dATHFISdyemFNM4nXvmwy0g7ICamPZ7CB/IU+sI4WwbiZS83jULObdwxNXvxwM1lrWOA06h6rgMPFiBn+qiMLtZXpkprUsDW2FNS69MOpODrf6kUIyRL/QXg9IJ08veQmhzEr8G7YQHsdz07b0Wmj/50nQPJAtdug/kh+RlGpYjjx8ALS6jGD2Gq; __snaker__id=RQbe8vWX2MrZo7OW; cmci9xde=U2FsdGVkX184glWsyX1mezYXZa3qDzIhjRBe9TQQVEY+LLFTFMDXP9nSs9RkgMzxPSZtU4lOrgeYZefNb02KEA==; pmck9xge=U2FsdGVkX1/T3lTFMBSW1MNHTt55yQ7uWCI2fzMhaBs=; assva6=U2FsdGVkX1803CTSuJgB3mtcX8vMBRJ4mrNXHxoktsA=; assva5=U2FsdGVkX18s7ilkblADh/oGQosbZLgtP7rzonbhC4T0Gf8YWm1GZf8atIJSu1QVH69xU8U+rNeMHgJRf8dEEQ==; vmce9xdq=U2FsdGVkX1+9L5kRV9p5NPBm2ZFxevxsXB601UTW1lO8oB1sywbxX35uCVgDtPFrCRoAhGz6Qw95IDt5HxZKgRgHcwII5jsliZ9AEeEMx0QyMg0NlFbFg2/No39rs8FcREB1wxx4Hg7ZLGq+HBSZ6UbnPSt1xTw3YsTv3I27GXM=; z_c0=2|1:0|10:1752912935|4:z_c0|92:Mi4xemV3ZkR3QUFBQUFBOEJFN19uN0FHU1lBQUFCZ0FsVk5KNkpvYVFCQlUzLUlBR3dQNGcwQkhEOWNXcmsyZ0VFZnJB|55035f8a3b883e94b09b952267d56c00a1a59a98cc43712feb238eb4056739b8; q_c1=e3e3832501fe4a17ba8bf7b5b47f6e60|1752912935000|1752912935000; __zse_ck=004_9njDuYcKusVXiMuD5SZ7LomPNAudnA3idCp0Ig/GY0=5gYSDmi0TNqJ7FjyXoqYArffzF1BhUy=GuO4ecVqJEGHl8QUnfJi3U5WYI/Y1QFhwF7Gz7I7xqjnm05LC1vY2-1G/2qpHnazSH76s+360GqWHjozS9rp18hQPVk+DOgjdq/n5leUgxJ+237tPuguqC5x1a1EjhuQ/RyoAp0+8lSPskVcy16jac+kELW9lwJayh1jscDZfo7NsPnlZfJUWL; gdxidpyhxdE=tdTqBn8olmH5j1Mvzgb2xuBP1JV%5CEOlNCeWjTWLag7PU1kAgwZc0iYoJNXHP4pesZ9c6pt6sCy9XvWRvcCih4H3wlMve85vPDsuwZGH0niB0eBEbOdwyCMBakjZYqhkOUjVurq3zUjP5bbW9MM0av5%2Fc9lNsShTacGcfNJ14hrRJYUyE%3A1752924069740; tst=h; SESSIONID=g7XUl1IInUF9AXT8Qhzu67qCg6YIsGbsJhzvVkXxO1E; JOID=V1kQAE26VE5LfxtUC7zz1qk37Fgd2WcQGxAuGUTXBQx0O0AEN73oNiV4G1YISQCL874KJt7dAvWPk3H3dDUAq3I=; osd=W1kVBkq2VEtNeBdUDrr02qky6l8R2WIWHBwuHELQCQxxPUcIN7juMSl4HlAPRQCO9bkGJtvbBfmPlnfweDUFrXU=; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1752950198; BEC=5ee33e0856ed13c879689106c041a08d"

def fetch_zhihu_hot(cookie_string: str = None):
    """
    严格复制指定的 fetch 请求，以获取知乎热榜页面。
    现在可以接受一个 cookie 字符串来模拟登录状态。
    """
    # 目标 URL
    url = "https://www.zhihu.com/hot"

    # 严格复制所有请求头
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "cache-control": "max-age=0",
        "priority": "u=0, i",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Google Chrome\";v=\"138\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    }

    # 解析 cookie 字符串为字典
    cookies = parse_cookie_string(cookie_string)

    # 使用 Session 对象可以自动处理 cookies
    session = requests.Session()

    try:
        # 发送 GET 请求，这次带上了 cookies 参数
        response = session.get(url, headers=headers, cookies=cookies)

        # 检查请求是否成功 (状态码 2xx)
        response.raise_for_status()

        print("请求成功!")
        print(f"状态码: {response.status_code}")
        # 打印响应内容的前500个字符以作验证
        print("响应内容 (前500字符):")
        print(response.text[:500])

        return response

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

def parse_cookie_string(cookie_string: str) -> Dict[str, str]:
    """将浏览器Cookie字符串解析成字典格式。"""
    cookies = {}
    if not cookie_string or cookie_string.lower() == 'your_cookie_string_here':
        return cookies
    for item in cookie_string.split(';'):
        item = item.strip()
        if '=' in item:
            key, value = item.split('=', 1)
            cookies[key] = value
    return cookies


# --- 同步获取回答评论函数 ---
def fetch_zhihu_answer_comments(answer_id: str, limit: int = 100) -> List[Dict]:
    # --- 此处代码保持不变，与原始功能一致 ---
    print(f"--- 开始获取回答ID: {answer_id} 的评论 (上限: {limit}条) ---")
    cookies = parse_cookie_string(ZHIHU_COOKIE_STRING)

    if not cookies:
        print("错误：全局变量 ZHIHU_COOKIE_STRING 为空或格式无效，无法进行请求。")
        return []

    api_url = f"https://www.zhihu.com/api/v4/answers/{answer_id}/comments"
    per_page_limit = 20
    params = {
        "include": "data[*].author,collapsed,reply_to_author,disliked,content,voting,vote_count,is_parent_author,is_author",
        "limit": per_page_limit,
        "offset": 0,
        "status": "open"
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": f"https://www.zhihu.com/question/123/answer/{answer_id}",
        "Accept": "application/json, text/plain, */*"
    }
    all_comments = []

    try:
        with requests.Session() as session:
            session.headers.update(headers)
            session.cookies.update(cookies)

            while len(all_comments) < limit:
                if len(all_comments) + per_page_limit > limit:
                    params['limit'] = limit - len(all_comments)

                response = session.get(api_url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                comments_data = data.get('data', [])
                if not comments_data:
                    print("信息：API未返回更多评论数据，已获取全部内容。")
                    break

                all_comments.extend(comments_data)
                print(f"成功获取 {len(comments_data)} 条评论，当前总数: {len(all_comments)}")

                paging_info = data.get('paging', {})
                if paging_info.get('is_end', True):
                    print("信息：API响应is_end=true，所有评论已加载完毕。")
                    break

                params['offset'] += len(comments_data)
                time.sleep(random.uniform(0.8, 2.0))

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 404:
            print(f"错误：获取评论失败，API返回404 Not Found。可能原因：回答ID({answer_id})无效或评论已关闭。")
        elif status_code in [401, 403]:
            print(f"错误：获取评论失败，API返回 {status_code} (Unauthorized/Forbidden)。请检查Cookie是否正确且未过期。")
        else:
            print(f"错误：HTTP请求失败，状态码: {status_code}，响应内容: {e.response.text}")
        return []
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"错误：发生网络请求或JSON解析错误: {e}")
        return all_comments[:limit]
    except Exception as e:
        print(f"未知错误: {e}")
        return []

    print(f"--- 任务完成，共获取 {len(all_comments)} 条评论 ---")
    return all_comments[:limit]


# --- 同步获取问题回答并补充评论 ---
def fetch_question_answers(question_id: str, output_filename: str, desired_answers: int = 5, max_no_increase: int = 3):
    print(f"--- 目标问题ID: {question_id}, 期望获取 {desired_answers} 个回答 ---")
    all_answers_data = []
    no_increase_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-infobars"],
            slow_mo=50
        )
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
            storage_state=AUTH_FILE,
        )
        context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN','en-US'] });
            window.chrome = { runtime: {} };
            """
        )

        page = context.new_page()

        def handle_response(response):
            if "/api/v4/questions/" in response.url and "/feeds" in response.url:
                try:
                    data = response.json()
                    new_answers = data.get('data', [])
                    print(f"--- 捕获到 {len(new_answers)} 条新回答的API响应 ---")
                    for new_answer in new_answers:
                        all_answers_data.append(new_answer.get('target', {}))
                except Exception as e:
                    print(f"解析API响应失败: {e}, URL: {response.url}")

        page.on("response", handle_response)

        try:
            question_url = f"https://www.zhihu.com/question/{question_id}"
            print(f">>> 正在访问问题页面: {question_url}")
            page.goto(question_url, wait_until="networkidle", timeout=60000)
            print(">>> 等待问题标题加载...")
            page.wait_for_selector('h1.QuestionHeader-title', timeout=15000)
            print(">>> 页面加载成功，问题标题已出现。")

            script_tag = page.query_selector('#js-initialData')
            if script_tag:
                json_data_str = script_tag.inner_text()
                initial_data = json.loads(json_data_str)
                initial_answers = initial_data.get('initialState', {}).get('entities', {}).get('answers', {})
                if initial_answers:
                    all_answers_data.extend(initial_answers.values())
                    print(f"--- 成功从初始HTML中提取 {len(initial_answers)} 个回答 ---")
            else:
                print("警告: 页面加载后未找到 #js-initialData 标签，可能无法获取第一页回答。")

            while len(all_answers_data) < desired_answers:
                prev_count = len(all_answers_data)
                page.evaluate("window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })")
                time.sleep(random.uniform(2.5, 4.0))
                curr_count = len(all_answers_data)
                if curr_count > prev_count:
                    no_increase_count = 0
                    print(f"回答数量增加：{prev_count} → {curr_count}，继续滚动。")
                else:
                    no_increase_count += 1
                    print(f"警告：滚动后回答数量未增加（{no_increase_count}/{max_no_increase}）。")
                    if no_increase_count >= max_no_increase:
                        print("连续多次未增，停止抓取。")
                        break

        except PlaywrightTimeoutError as e:
            print(f"在页面加载或操作过程中发生超时: {e}")
            page.screenshot(path="error_screenshot.png", full_page=True)
            print("已保存错误截图至 error_screenshot.png")
        except Exception as e:
            print(f"在页面加载或操作过程中发生错误: {e}")
            page.screenshot(path="error_screenshot.png", full_page=True)
            print("已保存错误截图至 error_screenshot.png")
        finally:
            browser.close()

    print(f"\n--- 抓取完成，共捕获 {len(all_answers_data)} 条回答数据 ---")
    final_data = all_answers_data[:desired_answers]
    save_json(output_filename, final_data)
    print(f"原始API响应数据已保存至文件: {output_filename}")

    print(f"\n--- 开始补充评论信息获取 ---")
    for answer in final_data:
        answer_id = answer.get("id")
        if not answer_id:
            print(f"警告: 跳过无效回答数据: {answer}")
            continue
        comments = fetch_zhihu_answer_comments(answer_id, limit=50)
        answer["comments"] = comments
        print(f"回答ID {answer_id} 的评论数量: {len(comments)}")
    save_json(output_filename, final_data)
    print(f"最终结果已保存至文件: {output_filename}")


if __name__ == "__main__":
    question_id = "1929972027688707067"
    fetch_question_answers(question_id, f"zhihu_answers_{question_id}.json", desired_answers=10)
