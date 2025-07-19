import time
from playwright.sync_api import sync_playwright

# --- 配置 ---
# 登录哪个网站
LOGIN_URL = "https://www.zhihu.com/signin"
# 保存认证状态的文件名
AUTH_FILE = "zhihu_auth_state.json"


def save_authentication_state():
    """
    启动浏览器，让用户手动登录，然后将浏览器的认证状态保存到文件中。
    """
    with sync_playwright() as p:
        # 启动一个非无头浏览器，这样我们才能看到界面并操作
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print(f"正在打开知乎登录页面: {LOGIN_URL}")
        page.goto(LOGIN_URL)

        print("\n" + "=" * 50)
        print("请在打开的浏览器窗口中手动完成登录操作。")
        print("登录成功后，请不要关闭浏览器，脚本将在一分钟后自动保存登录状态。")
        print("=" * 50 + "\n")

        # 留出 60 秒给用户进行登录操作
        # time.sleep(60)

        # 核心步骤：保存当前上下文的认证状态到指定文件
        context.storage_state(path=AUTH_FILE)

        print(f"成功！认证状态已保存到文件: {AUTH_FILE}")
        print("现在可以关闭这个脚本和浏览器了。")

        browser.close()


if __name__ == "__main__":
    save_authentication_state()