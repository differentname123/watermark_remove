import json
import os
import re
import time
import argparse
import sys
from typing import Tuple, Optional
from playwright.sync_api import sync_playwright, Page, expect
from playwright.sync_api import Page, expect, Locator

import time
import datetime
import sys
import csv
import os
import traceback # 用于捕获更详细的异常信息
# ==============================================================================
# 配置区域
# ==============================================================================
# 用于保存浏览器登录状态的目录，请确保该目录可写
# 第一次运行登录后，这里会生成包含cookies等信息的文件
USER_DATA_DIR = r"W:\temp\taobao2"
TARGET_URL_BASE = 'https://aistudio.google.com/prompts/new_chat'

# ==============================================================================
# 核心功能函数
# ==============================================================================


class PageCrashedException(Exception):
    """自定义异常，用于表示页面已崩溃。"""
    pass


def check_for_crash_and_abort(page: Page):
    """
    (内部调用) 快速检查页面是否崩溃。如果崩溃，则立即抛出异常以终止任务。
    """
    try:
        # 查找崩溃页面的特征元素：“重新加载”按钮。
        # 在简体中文环境下，按钮文本是 "重新加载"。
        reload_button = page.get_by_role("button", name="重新加载")

        # 使用极短的超时来检查，因为它应该立即存在于崩溃页面上。
        # 如果页面正常，这个检查会很快失败，不会浪费时间。
        if reload_button.is_visible(timeout=1000):  # 1秒超时
            error_msg = "页面已崩溃 (检测到 '重新加载' 按钮)，任务终止。"
            print(f"[!] {error_msg}")
            # 抛出自定义异常，这样我们可以在主逻辑中捕获它并进行处理。
            raise PageCrashedException(error_msg)

    except Exception as e:
        # 如果在1秒内找不到按钮 (抛出 TimeoutError)，或者发生其他错误，
        # 都意味着页面大概率是正常的，我们可以安全地忽略这个异常。
        # 我们只关心 PageCrashedException。
        if isinstance(e, PageCrashedException):
            raise  # 将我们自己的异常重新抛出
        # 其他异常（如 TimeoutError）则忽略
        pass


def login_and_save_session(model_name: str = "gemini-2.5-pro"):
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
            args=['--disable-blink-features=AutomationControlled', '--start-maximized', '--disable-gpu'],
            ignore_default_args=["--enable-automation"]
        )

        page = context.new_page()
        target_url = f"{TARGET_URL_BASE}?model={model_name}"
        page.goto(target_url)

        print("\n" + "="*60)
        print("浏览器已打开。请在浏览器窗口中手动完成登录操作。")
        print("登录成功并进入AI Studio主界面后，请回到本命令行窗口，然后按 Enter 键继续...")
        print("="*60)

        # 阻塞程序，等待用户在命令行按 Enter
        input()

        # 用户按 Enter 后，关闭浏览器，此时登录状态已自动保存到 USER_DATA_DIR
        context.close()
        print("\n[+] 登录会话信息已成功保存。现在可以使用 'query' 命令来运行任务了。")


def click_acknowledge_if_present(page: Page):
    """
    (内部调用) 检查并点击可能出现的 "Acknowledge" 弹窗按钮。
    此函数会快速检查按钮是否存在，如果不存在则不会等待，避免拖慢流程。
    """
    # print("[*] 正在检查 'Acknowledge' 弹窗...")
    time.sleep(2)

    # 使用 get_by_role 是 Playwright 推荐的最健壮的方式
    # 它会同时匹配按钮的可见文本 "Acknowledge"
    acknowledge_button = page.get_by_role("button", name="Acknowledge")

    try:
        # 使用一个非常短的超时时间来检查按钮是否可见
        # 如果弹窗存在，它通常会很快出现
        if acknowledge_button.is_visible(timeout=3000):  # 等待最多3秒
            print("[+] 检测到 'Acknowledge' 按钮，正在点击...")
            acknowledge_button.click()
            # 等待按钮消失，确认弹窗已关闭
            expect(acknowledge_button).to_be_hidden(timeout=5000)
            print("[+] 'Acknowledge' 弹窗已处理。")
        else:
            print("[-] 未发现 'Acknowledge' 弹窗，继续执行。")
            pass
    except Exception:
        # 如果在3秒内按钮没有出现，is_visible 会返回 False，不会抛出异常。
        # 这里加一个 except 以防万一，比如页面跳转导致检查失败。
        print("[-] 检查 'Acknowledge' 弹窗时发生意外或未找到，继续执行。")


def query_google_ai_studio(prompt: str, file_path: Optional[str] = None, user_data_dir=USER_DATA_DIR, model_name: str = "gemini-2.5-pro") -> Tuple[Optional[str], Optional[str]]:
    """
    使用已保存的登录会话启动浏览器，上传文件（可选），提交Prompt，并等待返回结果。

    Args:
        prompt (str): 提问的内容。
        user_data_dir (str): 保存浏览器登录状态的用户数据目录。
        file_path (str, optional): 附件文件的绝对路径。默认为 None。

    Returns:
        Tuple[str, str]: (error_info, response_text)
        - error_info: 如果出错，返回错误描述；否则为 None。
        - response_text: 如果成功，返回模型回答；否则为 None。
    """
    # 检查登录会话是否存在
    if not os.path.isdir(user_data_dir): # <-- 使用传入的参数
        error_msg = f"用户数据目录不存在: {user_data_dir}\n请先运行 'python {os.path.basename(__file__)} login --user-data-dir <你的目录>' 命令进行登录。"
        return error_msg, None

    error_info = None
    response_text = None
    context = None

    print(f"--- 开始任务: Prompt='{prompt[:20]}...', File='{file_path}' --- 当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. 检查文件路径（如果有）
        if file_path and not os.path.exists(file_path):
            raise FileNotFoundError(f"附件文件不存在: {file_path}")

        # 2. 启动 Playwright
        with sync_playwright() as p:
            try:
                # 启动持久化上下文，它会自动加载 user_data_dir 中的登录信息
                context = p.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir, # <-- 使用传入的参数
                    headless=False,  # 调试时建议开启 False，稳定后可改为 True
                    args=['--disable-blink-features=AutomationControlled', '--start-maximized', '--disable-gpu'],
                    ignore_default_args=["--enable-automation"]
                )
            except Exception as e:
                raise Exception(f"启动浏览器失败，请检查或确认浏览器是否已关闭: {e}")

            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(60000)  # 设置默认超时时间 60秒

            # 3. 访问页面
            print("[*] 正在加载页面...")
            target_url = f"{TARGET_URL_BASE}?model={model_name}"
            page.goto(target_url)
            # time.sleep(1000)
            # [修改] 页面加载后立即检查崩溃
            check_for_crash_and_abort(page)

            # 4. 处理弹窗和上传附件
            click_acknowledge_if_present(page)

            if file_path:
                # [修改] 操作前再次检查
                check_for_crash_and_abort(page)
                _upload_attachment(page, file_path)

            # [修改] 提交前也检查一下，确保页面状态良好
            check_for_crash_and_abort(page)

            for i in range(3):
                # 5. 提交 Prompt
                _submit_prompt(page, prompt)
                # 6. 等待并获取响应
                response_text = _wait_and_get_response(page)
                if "An internal error has occurred." not in response_text:
                    break
                time.sleep(2)
                print("[-] 检测到内部错误，正在重试...")
            print(f"[+] 任务成功任务完成任务。{file_path} {response_text[:100]}...")

    # [修改] 新增对页面崩溃异常的捕获
    except PageCrashedException as crash_e:
        error_info = str(crash_e)
        # 崩溃时截图
        if context and context.pages:
            try:
                screenshot_path = f"crash_screenshot_{int(time.time())}.png"
                # 确保有页面可以截图
                if context.pages:
                    context.pages[0].screenshot(path=screenshot_path)
                    print(f"[*] 崩溃截图已保存至: {screenshot_path}")
            except Exception as screenshot_e:
                print(f"[!] 截取崩溃快照失败: {screenshot_e}")

    except Exception as e:
        error_info = str(e)
        print(f"[!] 执行过程中发生错误: {error_info} {file_path}")
        # 可选：出错时截图
        if context and context.pages:
            try:
                screenshot_path = f"error_screenshot_{int(time.time())}.png"
                if context.pages:
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
    """(内部调用) 上传附件逻辑 (两步点击均已强化，具备高兼容性)"""
    print(f"[*] 正在上传附件: {os.path.basename(file_path)}")
    click_acknowledge_if_present(page)

    with page.expect_file_chooser(timeout=15000) as fc_info:
        # --- 第 1 步: 点击主附件按钮 (已强化) ---
        best_locator = page.locator('[data-test-add-chunk-menu-button]')
        fallback_locator = page.get_by_role(
            "button",
            name=re.compile(r"images, videos, files, or audio", re.IGNORECASE)
        )
        attachment_button = best_locator.or_(fallback_locator)
        attachment_button.click()

        # --- 第 2 步: 点击"上传文件"菜单项 (新增的强化) ---
        # 使用正则表达式来兼容 "Upload a file" 和 "Upload File" 等多种写法
        upload_option = page.get_by_role(
            "menuitem",
            name=re.compile(r"Upload (a )?file", re.IGNORECASE)
        )
        upload_option.click()

    file_chooser = fc_info.value
    file_chooser.set_files(file_path)

    spinner = page.locator(".upload-spinner")
    expect(spinner).to_be_hidden(timeout=60000)
    print("[+] 附件上传完毕。")


def _submit_prompt(page: Page, prompt: str):
    """(内部调用) 填写并提交Prompt (已升级为高兼容性定位器)"""
    print("[*] 正在提交Prompt...")

    # --- 1. 定位输入框 (替换旧的 get_by_placeholder) ---
    prompt_input: Locator | None = None
    try:
        # 步骤 A: 基础筛选 - 获取所有可见的 textbox 元素
        all_textboxes = page.get_by_role("textbox").filter(has_not_text="hidden").all()

        if not all_textboxes:
            raise Exception("在页面上找不到任何可见的输入框 (role='textbox')。")

        # 步骤 B: 位置筛选 - 过滤出位于页面下半部分的
        viewport_height = page.viewport_size['height']
        lower_half_textboxes = [
            box for box in all_textboxes
            if box.bounding_box()['y'] > viewport_height / 3
        ]

        if not lower_half_textboxes:
            raise Exception("在页面的下半部分找不到任何可见的输入框。")

        # 步骤 C: 智能决胜
        if len(lower_half_textboxes) == 1:
            prompt_input = lower_half_textboxes[0]
        else:
            # 优先选择 aria-label 包含关键意图词的
            tie_breaker_keywords = re.compile("prompt|type|enter|start typing", re.IGNORECASE)

            preferred_candidates = [
                box for box in lower_half_textboxes
                if tie_breaker_keywords.search(box.get_attribute("aria-label") or "")
            ]

            if len(preferred_candidates) == 1:
                prompt_input = preferred_candidates[0]
            else:
                # 备用策略：选择最后一个
                prompt_input = lower_half_textboxes[-1]

    except Exception as e:
        print(f"[!] 错误：定位Prompt输入框失败。 {e}")
        # 如果你想兼容旧版，可以在这里回退到旧的定位器
        print("[!] 尝试使用旧的 placeholder 定位器作为备用方案...")
        prompt_input = page.get_by_placeholder("Start typing a prompt")

    # --- 2. 填写Prompt (与原代码一致) ---
    expect(prompt_input).to_be_editable(timeout=15000)
    prompt_input.fill(prompt)

    # --- 3. 定位并点击运行按钮 (增加对 "Generate" 的兼容) ---
    # 使用正则表达式同时匹配 "Run" 和 "Generate"
    run_button = page.get_by_role(
        "button",
        name=re.compile(r"^(Run|Generate)$", re.IGNORECASE)
    )

    expect(run_button).to_be_enabled(timeout=300000)
    run_button.click()


def _scroll_page_to_bottom(page: Page, steps: int = 20, step_px: int = 1500, delay: float = 0.05):
    """不看任何容器，直接强制往下滚页面，保证滚到最底部"""
    for _ in range(steps):
        try:
            # 将鼠标移到视口中央
            vp = page.viewport_size or {"width": 1280, "height": 720}
            page.mouse.move(vp["width"]/2, vp["height"]/2)
            # 滑轮往下滑
            page.mouse.wheel(0, step_px)
        except:
            pass
        time.sleep(delay)
    # 最后用一次键盘 END 强制到底
    try:
        page.keyboard.press("End")
    except:
        pass

def _wait_and_get_response(page: Page) -> str:
    """(内部调用) 等待流式输出结束并提取文本"""
    print("[*] 等待模型响应中...")
    stop_btn = page.locator("button").filter(has_text="Stop")
    expect(stop_btn).to_be_visible(timeout=30000)
    expect(stop_btn).to_be_hidden(timeout=300000)
    _scroll_page_to_bottom(page, steps=40)  # 再滚到底，确保看到最后生成的节点
    time.sleep(1)  # 等待1秒，确保内容稳定
    response_container = page.locator('[data-turn-role="Model"]').last

    expect(response_container).to_be_visible()
    return response_container.inner_text()


# ==============================================================================
# ==============================================================================
# 程序主入口和使用示例
# ==============================================================================
if __name__ == '__main__':
    login_and_save_session()

    # 测试文件路径
    test_file = r"W:\project\python_project\watermark_remove\common_utils\video_scene\test.jpg"
    test_prompt = "请详细描述这张图片的内容。"

    # 调用封装好的函数
    err, response = query_google_ai_studio(prompt=test_prompt, file_path=test_file)

    if err:
        print("\n======== ❌ 失败 ========")
        print(f"错误信息: {err}")
    else:
        print("\n======== ✅ 成功 ========")
        print("模型回复内容:")
        print(response)