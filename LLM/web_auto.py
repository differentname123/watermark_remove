import os
import time
import argparse
import sys
from typing import Tuple, Optional
from playwright.sync_api import sync_playwright, Page, expect

# ==============================================================================
# 配置区域
# ==============================================================================
# 用于保存浏览器登录状态的目录，请确保该目录可写
# 第一次运行登录后，这里会生成包含cookies等信息的文件
USER_DATA_DIR = r"W:\temp\GoogleAIStudio_UserData"
TARGET_URL = 'https://aistudio.google.com/'

# ==============================================================================
# 核心功能函数
# ==============================================================================

def login_and_save_session():
    """
    启动浏览器，让用户手动登录，并将登录会话保存到 USER_DATA_DIR。
    """
    print("--- 启动浏览器进行手动登录 ---")
    print(f"会话信息将保存在: {USER_DATA_DIR}")

    with sync_playwright() as p:
        # 使用自带的 chromium，并启动持久化上下文
        context = p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=False,  # 必须为 False 以便用户可以看到和操作浏览器
            args=['--disable-blink-features=AutomationControlled', '--start-maximized'],
            ignore_default_args=["--enable-automation"]
        )

        page = context.new_page()
        page.goto(TARGET_URL)

        print("\n" + "="*60)
        print("浏览器已打开。请在浏览器窗口中手动完成登录操作。")
        print("登录成功并进入AI Studio主界面后，请回到本命令行窗口，然后按 Enter 键继续...")
        print("="*60)

        # 阻塞程序，等待用户在命令行按 Enter
        input()

        # 用户按 Enter 后，关闭浏览器，此时登录状态已自动保存到 USER_DATA_DIR
        context.close()
        print("\n[+] 登录会话信息已成功保存。现在可以使用 'query' 命令来运行任务了。")


def query_google_ai_studio(prompt: str, file_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    使用已保存的登录会话启动浏览器，上传文件（可选），提交Prompt，并等待返回结果。

    Args:
        prompt (str): 提问的内容。
        file_path (str, optional): 附件文件的绝对路径。默认为 None。

    Returns:
        Tuple[str, str]: (error_info, response_text)
        - error_info: 如果出错，返回错误描述；否则为 None。
        - response_text: 如果成功，返回模型回答；否则为 None。
    """
    # 检查登录会话是否存在
    if not os.path.isdir(USER_DATA_DIR):
        error_msg = f"用户数据目录不存在: {USER_DATA_DIR}\n请先运行 'python {os.path.basename(__file__)} login' 命令进行登录。"
        return error_msg, None

    error_info = None
    response_text = None
    context = None

    print(f"--- 开始任务: Prompt='{prompt[:20]}...', File='{file_path}' ---")

    try:
        # 1. 检查文件路径（如果有）
        if file_path and not os.path.exists(file_path):
            raise FileNotFoundError(f"附件文件不存在: {file_path}")

        # 2. 启动 Playwright
        with sync_playwright() as p:
            try:
                # 启动持久化上下文，它会自动加载 USER_DATA_DIR 中的登录信息
                context = p.chromium.launch_persistent_context(
                    user_data_dir=USER_DATA_DIR,
                    headless=False,  # 调试时建议开启 False，稳定后可改为 True
                    args=['--disable-blink-features=AutomationControlled', '--start-maximized'],
                    ignore_default_args=["--enable-automation"]
                )
            except Exception as e:
                raise Exception(f"启动浏览器失败，请检查或确认浏览器是否已关闭: {e}")

            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(60000)  # 设置默认超时时间 60秒

            # 3. 访问页面
            print("[*] 正在加载页面...")
            page.goto(TARGET_URL)

            # 4. 上传附件 (如果存在)
            if file_path:
                _upload_attachment(page, file_path)

            # 5. 提交 Prompt
            _submit_prompt(page, prompt)

            # 6. 等待并获取响应
            response_text = _wait_and_get_response(page)
            print("[+] 任务成功完成。")

    except Exception as e:
        error_info = str(e)
        print(f"[!] 执行过程中发生错误: {error_info}")
        # 可选：出错时截图
        if context and context.pages:
            try:
                screenshot_path = f"error_screenshot_{int(time.time())}.png"
                context.pages[0].screenshot(path=screenshot_path)
                print(f"[*] 错误截图已保存至: {screenshot_path}")
            except Exception as screenshot_e:
                print(f"[!] 截取错误快照失败: {screenshot_e}")

    finally:
        # 7. 清理资源
        if context:
            try:
                context.close()
                print("[*] 浏览器环境已关闭。")
            except Exception:
                pass

    return error_info, response_text


# ==============================================================================
# 内部辅助函数 (与原版相同，无需修改)
# ==============================================================================

def _upload_attachment(page: Page, file_path: str):
    """(内部调用) 上传附件逻辑"""
    print(f"[*] 正在上传附件: {os.path.basename(file_path)}")
    with page.expect_file_chooser(timeout=15000) as fc_info:
        attachment_button = page.locator('[aria-label="Insert images, videos, audio, or files"]')
        attachment_button.click()
        upload_option = page.get_by_role("menuitem", name="Upload a file")
        upload_option.click()
    file_chooser = fc_info.value
    file_chooser.set_files(file_path)
    spinner = page.locator(".upload-spinner")
    expect(spinner).to_be_hidden(timeout=60000)
    print("[+] 附件上传完毕。")


def _submit_prompt(page: Page, prompt: str):
    """(内部调用) 填写并提交Prompt"""
    print("[*] 正在提交Prompt...")
    prompt_input = page.get_by_placeholder("Start typing a prompt")
    expect(prompt_input).to_be_editable(timeout=15000)
    prompt_input.fill(prompt)
    run_button = page.get_by_role("button", name="Run", exact=True)
    expect(run_button).to_be_enabled(timeout=15000)
    run_button.click()


def _wait_and_get_response(page: Page) -> str:
    """(内部调用) 等待流式输出结束并提取文本"""
    print("[*] 等待模型响应中...")
    stop_btn = page.locator("button").filter(has_text="Stop")
    expect(stop_btn).to_be_visible(timeout=30000)
    expect(stop_btn).to_be_hidden(timeout=300000)
    response_container = page.locator('[data-turn-role="Model"]').last
    expect(response_container).to_be_visible()
    return response_container.inner_text()


# ==============================================================================
# 程序主入口和使用示例
# ==============================================================================

if __name__ == '__main__':

    # login_and_save_session()

    # --------- 在这里配置你的查询任务 ---------
    test_file = r"W:\project\python_project\watermark_remove\common_utils\video_scene\test.jpg"
    test_prompt = "请详细描述这张图片的内容。"
    # ----------------------------------------

    # 调用封装好的函数
    err, response = query_google_ai_studio(prompt=test_prompt, file_path=test_file)

    if err:
        print("\n======== ❌ 失败 ========")
        print(f"错误信息: {err}")
        sys.exit(1) # 脚本以错误码退出
    else:
        print("\n======== ✅ 成功 ========")
        print("模型回复内容:")
        print(response)