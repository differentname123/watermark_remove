import time
import pandas as pd
import random
import os
import sys
import logging
from playwright.sync_api import sync_playwright

# --- 配置区域 ---
AUTH_FILE = "zhihu_auth_state.json"  # 登录态文件
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.5790.110 Safari/537.36"
)
PROXY_SERVER = None  # 如 "http://user:pass@proxy.host:3128"，不需要可留 None
SELECTORS = {
    "item_container": 'section[data-za-detail-view-path-module="FeedItem"]',
    "rank": "div.HotItem-rank",
    "title": "h2.HotItem-title",
    "metrics": "div.HotItem-metrics",
    "link": "div.HotItem-content > a"
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def init_browser_context(headless: bool = True):
    """
    初始化并返回 (browser, context, page)。
    包含 headless 设置、代理、伪装脚本注入等。
    """
    playwright = sync_playwright().start()
    launch_args = [
        "--headless=new" if headless else None,
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--disable-infobars"
    ]
    if PROXY_SERVER:
        launch_args.append(f"--proxy-server={PROXY_SERVER}")
    launch_args = [arg for arg in launch_args if arg]

    browser = playwright.chromium.launch(
        headless=headless,
        args=launch_args,
        slow_mo=50
    )
    context = browser.new_context(
        user_agent=USER_AGENT,
        viewport={"width": 1920, "height": 1080},
        locale="zh-CN",
        storage_state=AUTH_FILE,
    )
    context.add_init_script("""
      Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
      Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
      Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN','en-US'] });
      window.chrome = { runtime: {} };
      const toDataURL = HTMLCanvasElement.prototype.toDataURL;
      HTMLCanvasElement.prototype.toDataURL = function() {
        const ctx = this.getContext('2d');
        const img = ctx.getImageData(0, 0, this.width, this.height);
        for (let i = 0; i < img.data.length; i += 4) {
          img.data[i]   += (Math.random() - 0.5);
          img.data[i+1] += (Math.random() - 0.5);
        }
        ctx.putImageData(img, 0, 0);
        return toDataURL.apply(this, arguments);
      };
    """)
    page = context.new_page()
    return playwright, browser, context, page


def human_mouse_movements(page, box):
    """模拟随机鼠标轨迹到目标元素中心，并短暂停留。"""
    cx = box['x'] + box['width'] / 2 + random.randint(-5, 5)
    cy = box['y'] + box['height'] / 2 + random.randint(-5, 5)
    steps = random.randint(8, 20)
    page.mouse.move(cx, cy, steps=steps)
    time.sleep(random.uniform(0.1, 0.4))


def get_zhihu_hot_list():
    if not os.path.exists(AUTH_FILE):
        logging.error(f"认证文件 '{AUTH_FILE}' 未找到，请先执行保存登录状态脚本。")
        sys.exit(1)

    playwright, browser, context, page = init_browser_context(headless=False)
    logging.info("浏览器启动并加载登录态。")

    try:
        logging.info("导航至 https://www.zhihu.com/hot …")
        page.goto("https://www.zhihu.com/hot", timeout=60000)
        page.wait_for_selector(SELECTORS["item_container"], timeout=30000)

        logging.info("开始滚动加载所有内容…")
        last_h = page.evaluate("document.body.scrollHeight")
        for _ in range(random.randint(4, 7)):
            page.mouse.wheel(0, last_h // random.randint(5, 10))
            time.sleep(random.uniform(1.5, 3.5))
            new_h = page.evaluate("document.body.scrollHeight")
            if new_h == last_h:
                break
            last_h = new_h
        time.sleep(random.uniform(2.0, 4.0))

        logging.info("解析热榜条目…")
        items = page.query_selector_all(SELECTORS["item_container"])
        result = []
        for item in items:
            box = item.bounding_box()
            human_mouse_movements(page, box)
            page.hover(SELECTORS["title"])
            time.sleep(random.uniform(0.2, 0.6))

            rank_el = item.query_selector(SELECTORS["rank"])
            title_el = item.query_selector(SELECTORS["title"])
            metrics_el = item.query_selector(SELECTORS["metrics"])
            link_el = item.query_selector(SELECTORS["link"])

            title = title_el.inner_text().strip() if title_el else None
            if not title:
                continue
            href = link_el.get_attribute("href") if link_el else ""
            href = href if href.startswith("http") else f"https://www.zhihu.com{href}"

            result.append({
                "排名": rank_el.inner_text().strip() if rank_el else "N/A",
                "标题": title,
                "热度": metrics_el.inner_text().strip() if metrics_el else "N/A",
                "链接": href
            })
            time.sleep(random.uniform(0.1, 0.3))

        if len(result) < 10:
            logging.warning(f"仅抓取到 {len(result)} 条，可能触发风控。")
        else:
            df = pd.DataFrame(result)
            df.to_csv("zhihu_hot_list.csv", index=False, encoding="utf-8-sig")
            logging.info(f"抓取成功，共 {len(result)} 条记录。")
            print(df.head())

    except Exception as e:
        logging.error("运行出错：", exc_info=True)
        try:
            if page.locator("div.Captcha-container").count() > 0:
                logging.warning("检测到人机验证页面！")
        except:
            pass
        page.screenshot(path="error_screenshot.png", full_page=True)
        logging.info("已保存错误截图：error_screenshot.png")
    finally:
        context.close()
        browser.close()
        playwright.stop()


if __name__ == "__main__":
    get_zhihu_hot_list()
