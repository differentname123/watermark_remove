# multiprocess_runner.py
import concurrent.futures
import multiprocessing
from typing import Dict, Any
# 先: pip install playwright
# 然后: playwright install
# 推荐: 本地有 Chrome/Edge 安装，或者用 Playwright 的 channel="chrome"
from urllib.parse import urlparse # <--- 引入 urlparse

import time
import random
from typing import List, Dict, Optional
from playwright.sync_api import sync_playwright, TimeoutError, Error

from common_utils.common_utils import get_config, init_config


def _parse_cookie_string(cookie_string: str) -> Dict[str, str]:
    cookies = {}
    for part in cookie_string.split(';'):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        cookies[k.strip()] = v.strip()
    return cookies

def visit_bilibili_with_cookies_fix(
    cookie_string: str,
    video_url: str = 'https://www.bilibili.com/video/BV16Sa3zhEPe/',
    headless: bool = False,                # 默认不开 headless，尽量像真实浏览器
    watch_seconds: int = 60,
    use_system_chrome: bool = True,        # 尝试使用系统 Chrome（更可靠）
    chrome_executable_path: Optional[str] = None,  # 如果 channel 不可用，可传 chrome 可执行路径
):
    """
    改进版：尽量解决 '浏览器不支持 HTML5 播放器' 的提示。

    参数:
      cookie_string: "k1=v1; k2=v2; ..." 的 cookie 字符串（最好包含 SESSDATA）
      video_url: 目标视频
      headless: 是否无头（建议 False）
      watch_seconds: 最多观看多少秒
      use_system_chrome: 是否尝试用系统 Chrome（优先）
      chrome_executable_path: 可选，显式指定 Chrome 可执行文件路径（例如 C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe）
    返回:
      dict 运行结果
    """
    cookies_dict = _parse_cookie_string(cookie_string)
    result = {
        "likely_logged_in_from_cookie": 'SESSDATA' in cookies_dict or 'bili_jct' in cookies_dict,
        "page_video_found": False,
        "played_video": False,
        "page_login_status_guess": None,
        "error": None,
    }

    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/116.0.0.0 Safari/537.36"
    )

    # 转换 cookie 到 Playwright 格式
    pk_cookies: List[dict] = []
    for k, v in cookies_dict.items():
        pk_cookies.append({
            "name": k,
            "value": v,
            "domain": ".bilibili.com",
            "path": "/",
        })

    p = None
    browser = None
    try:
        p = sync_playwright().start()

        launch_kwargs = dict(
            headless=headless,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--autoplay-policy=no-user-gesture-required",  # 允许自动播放
                "--disable-dev-shm-usage",
                "--disable-infobars",
                # 可按需加代理参数等
            ],
        )

        # 尝试优先用系统 Chrome（更可能支持 H264 等 codec）
        try:
            if use_system_chrome:
                # 尝试通过 channel（如果 Playwright 能找到 Chrome）
                browser = p.chromium.launch(channel="chrome", **launch_kwargs)
            else:
                if chrome_executable_path:
                    browser = p.chromium.launch(executable_path=chrome_executable_path, **launch_kwargs)
                else:
                    browser = p.chromium.launch(**launch_kwargs)
        except Exception as e_channel:
            # 如果 channel 不可用，尝试用显式可执行路径（若提供）
            if chrome_executable_path:
                browser = p.chromium.launch(executable_path=chrome_executable_path, **launch_kwargs)
            else:
                # 回退到默认 Playwright Chromium（但仍尽可能伪装）
                browser = p.chromium.launch(**launch_kwargs)

        context = browser.new_context(
            user_agent=user_agent,
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            viewport={"width": 1366, "height": 768},
            java_script_enabled=True,
        )

        # 注入 cookies
        if pk_cookies:
            context.add_cookies(pk_cookies)

        page = context.new_page()

        # 尽量隐藏 webdriver 指纹
        try:
            page.add_init_script("""() => {
                // Hide webdriver flag
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN','zh']});
                Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
                // Mock chrome object
                window.chrome = { runtime: {} };
            }""")
        except Exception:
            pass

        # 访问页面
        page.goto(video_url, wait_until="networkidle", timeout=45000)

        # 检查页面是否展示“不支持 HTML5”的提示（通过页面文本）
        try:
            # 如果页面里出现类似提示，尝试点击“切换为 HTML5 播放器”或调整 UA/重载
            if page.locator("text=不支持 HTML5").count() > 0 or page.locator("text=HTML5 播放器").count() > 0:
                result["error"] = "页面报告不支持 HTML5 播放器（检测到提示文本）。已尝试自动修复策略。"
                # 尝试寻找并点击切换按钮（若存在）
                possible_buttons = [
                    "text=切换为 HTML5 播放器",
                    "text=使用 HTML5 播放器",
                    "text=我知道了",
                ]
                for b in possible_buttons:
                    try:
                        if page.locator(b).count() > 0:
                            page.locator(b).first.click(timeout=3000)
                            time.sleep(1)
                    except Exception:
                        pass
                # 重新导航一次（让播放器重试）
                page.reload(wait_until="networkidle", timeout=30000)
        except Exception:
            pass

        # 等待 video 元素 或 帧内播放器
        video_el = None
        try:
            video_el = page.wait_for_selector("video", timeout=15000)
            result["page_video_found"] = video_el is not None
        except TimeoutError:
            result["page_video_found"] = False

        # 尝试播放
        played = False
        if result["page_video_found"]:
            try:
                played = page.evaluate("""() => {
                    const v = document.querySelector('video');
                    if (!v) return false;
                    try { v.muted = false; } catch(e){}
                    try { v.playbackRate = 1.0; } catch(e){}
                    return v.play().then(()=>true).catch(()=>false);
                }""")
            except Exception:
                played = False

        # 尝试通过点击播放按钮（备用）
        if not played:
            try:
                # 一些选择器在播放器覆盖层上
                click_selectors = [
                    ".bpx-player-ctrl .bpx-player-ctrl-play",
                    ".bilibili-player-video-btn",
                    ".play-btn",
                    "button[aria-label='播放']",
                    ".player-controller .play",
                ]
                for sel in click_selectors:
                    try:
                        loc = page.locator(sel)
                        if loc.count() > 0:
                            loc.first.click(timeout=3000)
                            time.sleep(0.8)
                            played = True
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        result["played_video"] = bool(played)

        # 做简单的人类行为模拟（滚动、移动鼠标）
        start = time.time()
        elapsed = 0
        while elapsed < watch_seconds:
            time.sleep(random.uniform(1.0, 2.5))
            try:
                page.evaluate(
                    """() => {
                        const h = document.documentElement.scrollHeight || document.body.scrollHeight;
                        const y = Math.floor(Math.random() * (h / 3)) + 50;
                        window.scrollTo({ top: y, behavior: 'smooth' });
                    }"""
                )
            except Exception:
                pass

            try:
                v = page.query_selector("video")
                if v:
                    box = v.bounding_box()
                    if box:
                        x = box["x"] + random.uniform(10, max(10, box["width"] - 10))
                        y = box["y"] + random.uniform(10, max(10, box["height"] - 10))
                        page.mouse.move(x, y, steps=random.randint(3, 12))
            except Exception:
                # 随机移动
                page.mouse.move(random.uniform(100, 1200), random.uniform(100, 600), steps=5)

            elapsed = time.time() - start

        # 最后抓取一些信息
        try:
            result["final_url"] = page.url
            result["page_title"] = page.title()
            # 再次简单判断登录状态
            try:
                result["page_login_status_guess"] = "maybe_not_logged_in" if page.locator("text=登录").count() > 0 else "maybe_logged_in"
            except Exception:
                result["page_login_status_guess"] = "unknown"
        except Exception:
            pass

    except Exception as e:
        result["error"] = str(e)
    finally:
        try:
            if browser:
                browser.close()
        except Exception:
            pass
        try:
            if p:
                p.stop()
        except Exception:
            pass

    return result


def refresh_page_periodically(
        url: str,
        watch_seconds: int = 60,
        cookie_string: Optional[str] = None,
        headless: bool = False,
        use_system_chrome: bool = True,
        chrome_executable_path: Optional[str] = None,
):
    """
    使用 Playwright 访问指定页面，并在指定时间内，每隔5秒刷新一次。
    (已修复刷新卡住的问题)
    """
    p = None
    browser = None
    try:
        p = sync_playwright().start()

        launch_kwargs = dict(
            headless=headless,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--autoplay-policy=no-user-gesture-required",
                "--disable-dev-shm-usage",
                "--disable-infobars",
            ],
        )

        try:
            if use_system_chrome:
                browser = p.chromium.launch(channel="chrome", **launch_kwargs)
            elif chrome_executable_path:
                browser = p.chromium.launch(executable_path=chrome_executable_path, **launch_kwargs)
            else:
                browser = p.chromium.launch(**launch_kwargs)
        except Exception:
            browser = p.chromium.launch(**launch_kwargs)

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/116.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            viewport={"width": 1366, "height": 768},
            java_script_enabled=True,
        )

        if cookie_string:
            cookies_dict = _parse_cookie_string(cookie_string)
            if cookies_dict:
                parsed_url = urlparse(url)
                hostname_parts = parsed_url.hostname.split('.')
                cookie_domain = f".{'.'.join(hostname_parts[-2:])}" if len(hostname_parts) > 1 else parsed_url.hostname

                pk_cookies: List[dict] = []
                for k, v in cookies_dict.items():
                    pk_cookies.append({
                        "name": k, "value": v, "domain": cookie_domain, "path": "/"
                    })
                context.add_cookies(pk_cookies)

        page = context.new_page()

        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        # --- 修改点 1：首次访问页面时，使用 'domcontentloaded' ---
        print(f"正在访问页面: {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=45000)  # <--- 主要修改点
        print("页面访问成功，开始刷新循环...")

        start_time = time.time()
        while time.time() - start_time < watch_seconds:
            elapsed_time = time.time() - start_time
            remaining_time = watch_seconds - elapsed_time

            sleep_duration = min(10.0, remaining_time)
            if sleep_duration <= 0:
                break

            print(f"等待 {sleep_duration:.1f} 秒后刷新... (总进度: {int(elapsed_time)}/{watch_seconds} 秒)")
            time.sleep(sleep_duration)

            # --- 修改点 2：刷新页面时，同样使用 'domcontentloaded' ---
            try:
                print("正在刷新页面...")
                page.reload(wait_until="domcontentloaded", timeout=30000)  # <--- 主要修改点
                print("刷新完成。")
            except Exception as e:
                print(f"刷新失败，尝试重新导航: {e}")
                try:
                    # --- 修改点 3：备用导航方案也使用 'domcontentloaded' ---
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)  # <--- 主要修改点
                except Exception as e2:
                    print(f"重新导航也失败了: {e2}")
                    break

        print(f"已达到指定的 {watch_seconds} 秒，任务结束。")

    except Exception as e:
        print(f"在执行过程中发生错误: {e}")
        raise
    finally:
        if browser:
            browser.close()
        if p:
            p.stop()
        print("浏览器和Playwright实例已关闭。")

def _worker_call(cookie_string: str, video_url: str, headless: bool, watch_seconds: int) -> Dict[str, Any]:
    """在子进程中运行的函数（必须为顶层函数以便 pickling）。"""
    try:
        res = refresh_page_periodically(
            cookie_string=cookie_string,
            url=video_url,
            headless=headless,
            watch_seconds=watch_seconds
        )
        # 不打印完整 cookie，返回一个简短提示
        return res
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # 要处理的 config key 列表（替换为你真实的 key）
    cookie_keys = ["nana_bilibili_total_cookie", "nana_bilibili_total_cookie", "dahao_bilibili_total_cookie",
                   "dan_bilibili_total_cookie", "mama_bilibili_total_cookie", "chabian_bilibili_total_cookie"]
    cookie_keys = ["mama_bilibili_total_cookie"]




    cookie_keys = []
    config = init_config()
    for key, value in config.items():
        if 'junda' != value['name']:
            continue
        cookie_keys.append(value['total_cookie'])


    video_url = "https://www.bilibili.com/video/BV11pa5z1EEy/"
    headless = False
    watch_seconds = 6000
    max_workers = len(cookie_keys)  # 根据机器调整

    # Windows 下推荐 'spawn'
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_worker_call, key, video_url, headless, watch_seconds): key for key in cookie_keys}
        for fut in concurrent.futures.as_completed(futures):
            key = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"key": key, "error": str(e)}
            results.append(r)

    # 打印每个任务结果
    for r in results:
        print(r)