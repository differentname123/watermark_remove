import os
import time
import base64
import traceback

# 假设 common_utils 是您本地的模块
from common_utils.common_utils import get_config
from PIL import Image
import functools
import json
import threading
from filelock import FileLock  # 新增：导入文件锁库

# 新增：用于识别图片文件类型
import mimetypes

# # 设置代理环境变量
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import google.genai as genai
from google.genai import types


# === 修改开始：引入健壮的、支持并发的 ApiKeyManager ===

class ApiKeyManager:
    """
    一个线程安全和进程安全的API密钥管理器。
    - 每次请求时动态排序密钥。
    - 使用文件锁来处理并发读写。
    """

    def __init__(self, api_key_map):
        self.api_key_map = api_key_map

        # --- 这是唯一的、核心的修改 ---
        # 获取当前文件（即此模块文件）所在的目录的绝对路径
        # 这样可以确保无论从哪个脚本调用，生成的文件都在这个模块旁边
        module_dir = os.path.dirname(os.path.abspath(__file__))

        # 将文件名与该目录拼接，形成一个固定的、绝对的路径
        self.stats_file = os.path.join(module_dir, 'api_key_usage.json')
        # --- 修改结束 ---

        self.lock_file = self.stats_file + '.lock'
        self.lock = FileLock(self.lock_file, timeout=10)
        self._initialize_stats()

    def _initialize_stats(self):
        """初始化统计文件，如果不存在或为空。"""
        with self.lock:
            if not os.path.exists(self.stats_file) or os.path.getsize(self.stats_file) == 0:
                initial_stats = {key: 0 for key in self.api_key_map.keys()}
                with open(self.stats_file, 'w') as f:
                    json.dump(initial_stats, f, indent=4)

    def get_ordered_keys(self):
        """
        【核心】获取根据使用次数动态排序的密钥名称列表。
        此操作是线程和进程安全的。
        """
        with self.lock:
            try:
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
                # 确保所有当前的 key 都在统计数据中
                for key in self.api_key_map.keys():
                    if key not in stats:
                        stats[key] = 0
            except (FileNotFoundError, json.JSONDecodeError):
                stats = {key: 0 for key in self.api_key_map.keys()}

        # 根据使用次数（值）对密钥（键）进行排序
        sorted_keys = sorted(stats.keys(), key=lambda k: stats.get(k, 0))
        print(f"[INFO] API 密钥将按以下动态顺序尝试: {sorted_keys}")
        return sorted_keys

    def record_success(self, key_name):
        """
        【核心】为一个成功的API调用记录次数。
        此操作是线程和进程安全的“读取-修改-写入”原子操作。
        """
        with self.lock:
            # 1. 再次读取最新数据，防止在等待锁期间文件已被其他进程修改
            try:
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                stats = {key: 0 for key in self.api_key_map.keys()}

            # 2. 修改数据
            stats[key_name] = stats.get(key_name, 0) + 1

            # 3. 写回文件
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            print(f"[INFO] 密钥 '{key_name}' 使用次数已更新为: {stats[key_name]}")


# 获取 API 密钥
API_KEY_BASE = get_config("gemini_api_key")
API_KEY_JIE = get_config("gemini_api_key_jie")
API_KEY_GE = get_config("gemini_api_key_ge")
API_KEY_NA = get_config("gemini_api_key_na")
API_KEY_CHU = get_config("gemini_api_key_chu")
API_KEY_RU = get_config("gemini_api_key_ru")
API_KEY_chu1 = get_config("gemini_api_key_chu1")
API_KEY_chu2 = get_config("gemini_api_key_chu2")
# API_KEY_chu3 = get_config("gemini_api_key_chu3")
API_KEY_chu4 = get_config("gemini_api_key_chu4")
API_KEY_chu5 = get_config("gemini_api_key_chu5")
# API_KEY_chu6 = get_config("gemini_api_key_chu6")
API_KEY_chu7 = get_config("gemini_api_key_chu7")
API_KEY_chu8 = get_config("gemini_api_key_chu8")
API_KEY_chu9 = get_config("gemini_api_key_chu9")
API_KEY_chu10 = get_config("gemini_api_key_chu10")
API_KEY_chu11 = get_config("gemini_api_key_chu11")
API_KEY_chu12 = get_config("gemini_api_key_chu12")
API_KEY_chu13 = get_config("gemini_api_key_chu13")
API_KEY_chu14 = get_config("gemini_api_key_chu14")
API_KEY_chu15 = get_config("gemini_api_key_chu15")
API_KEY_chu16 = get_config("gemini_api_key_chu16")
API_KEY_chu17 = get_config("gemini_api_key_chu17")




API_KEY_MAP = {
    'base':API_KEY_BASE,
    'jie':API_KEY_JIE,
    'ge':API_KEY_GE,
    'na':API_KEY_NA,
    'chu':API_KEY_CHU,
    'ru':API_KEY_RU,
    'chu1':API_KEY_chu1,
    'chu2': API_KEY_chu2,
    # 'chu3': API_KEY_chu3,
    # 'chu4': API_KEY_chu4,
    'chu5': API_KEY_chu5,
    # 'chu6': API_KEY_chu6,
    'chu7': API_KEY_chu7,
    'chu8': API_KEY_chu8,
    'chu9': API_KEY_chu9,
    'chu10': API_KEY_chu10,
    'chu11': API_KEY_chu11,
    'chu12': API_KEY_chu12,
    'chu13': API_KEY_chu13,
    'chu14': API_KEY_chu14,
    'chu15': API_KEY_chu15,
    'chu16': API_KEY_chu16,
    'chu17': API_KEY_chu17,

}

# 创建一个全局的 Key Manager 实例
# 所有的函数都将通过这个实例来获取密钥和记录成功
api_key_manager = ApiKeyManager(API_KEY_MAP)

# === 修改结束 ===


import google.generativeai as genai_flash
from google.api_core import exceptions as ga_exceptions


def with_proxy(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        try:
            return func(*args, **kwargs)
        finally:
            if 'HTTP_PROXY' in os.environ: del os.environ['HTTP_PROXY']
            if 'HTTPS_PROXY' in os.environ: del os.environ['HTTPS_PROXY']

    return wrapper

@with_proxy
def analyze_videos_gemini(
    prompt: str = '视频中的内容是什么',
    video_paths: list[str] = [],
    timeout: int = 600
) -> str:
    """
    针对一组视频文件，调用 Gemini-2.5-Pro 分析其内容并返回合并后的文本回复。

    参数：
    - prompt: 你希望模型在视频内容上回答的问题。
    - video_paths: 本地视频文件路径列表。
    - timeout: 单次请求的超时时间（秒）。
    """
    if video_paths is None:
        video_paths = []

    last_error = None
    ordered_keys = api_key_manager.get_ordered_keys()

    for key_name in ordered_keys:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key:
            continue

        genai_flash.configure(api_key=api_key)
        uploaded = []  # 存储 (原始文件名, 上传后的 video_file 对象)

        try:
            # 1. 上传并等待所有视频处理完成
            for path in video_paths:
                if not os.path.exists(path):
                    return f"错误: 视频文件未找到 -> {path}"
                basename = os.path.basename(path)
                print(f"[INFO] 使用 API Key “{key_name}” 上传视频 {basename} …")
                video_file = genai_flash.upload_file(path=path)
                # 等待处理完成
                while video_file.state.name == "PROCESSING":
                    print(f"[INFO] 视频 {basename} 正在处理…")
                    time.sleep(10)
                    video_file = genai_flash.get_file(video_file.name)
                if video_file.state.name == "FAILED":
                    raise RuntimeError(f"视频处理失败：{basename}")
                uploaded.append((basename, video_file))

            # 2. 构造 prompt_parts，包含“文件名:”提示
            prompt_parts = [
                prompt,
                "下面我将以 '文件名:' 的格式，在每个视频前提供其名称，请据此作答。"
            ]
            for basename, vf in uploaded:
                prompt_parts.append(f"{basename}:")  # 使用本地原始文件名
                prompt_parts.append(vf)             # 插入视频对象

            # 3. 调用 Gemini-2.5-Pro 生成内容
            print(f"[INFO] 使用 API Key “{key_name}” 调用 Gemini 模型生成内容…")
            model = genai_flash.GenerativeModel(model_name="gemini-2.5-pro")
            response = model.generate_content(
                prompt_parts,
                request_options={"timeout": timeout}
            )

            # 4. 记录成功并清理上传文件
            api_key_manager.record_success(key_name)
            for _, vf in uploaded:
                try:
                    print(f"[INFO] 删除临时文件 {vf.name} …")
                    genai_flash.delete_file(vf.name)
                except Exception as de:
                    print(f"[WARN] 删除视频文件 {vf.name} 失败：{de}")

            return response.text

        except (ga_exceptions.PermissionDenied,
                ga_exceptions.ResourceExhausted,
                ga_exceptions.GoogleAPICallError) as e:
            # 某些 Key 遇到配额或权限问题，切换到下一个
            last_error = e
            print(f"[WARN] API Key “{key_name}” 调用失败：{e}，尝试下一个…")
            # 删除已上传的文件，避免残留
            for _, vf in uploaded:
                try:
                    genai_flash.delete_file(vf.name)
                except:
                    pass
            continue

        except Exception as e:
            # 未知错误，直接返回
            return f"处理过程中发生未知错误: {e.__class__.__name__}: {e}"

    return f"所有 API Key 均尝试失败。最后一次错误：{last_error}"


@with_proxy
def get_llm_content_gemini_flash_video(
    prompt: str = '视频中的内容是什么',
    video_path: str = 'test.mp4',
    model_name: str = "gemini-2.5-flash"
) -> str:
    last_error = None
    ordered_keys = api_key_manager.get_ordered_keys()

    for key_name in ordered_keys:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key:
            continue

        genai_flash.configure(api_key=api_key)
        if not os.path.exists(video_path):
            return f"错误: 视频文件未找到 -> {video_path}"

        video_file = None
        try:
            print(f"[INFO] 使用 API Key “{key_name}” 上传视频…")
            api_key_manager.record_success(key_name)
            video_file = genai_flash.upload_file(path=video_path)

            # 等待处理
            while video_file.state.name == "PROCESSING":
                print("等待视频处理完成…")
                time.sleep(10)
                video_file = genai_flash.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise RuntimeError(f"视频处理失败：{video_path}")

            model = genai_flash.GenerativeModel(model_name=model_name)
            response = model.generate_content(
                [video_file, prompt],
                request_options={"timeout": 600}
            )

            return response.text

        except (ga_exceptions.PermissionDenied,
                ga_exceptions.ResourceExhausted,
                ga_exceptions.GoogleAPICallError) as e:
            # traceback.print_exc()

            last_error = e
            print(f"[WARN] Key “{key_name}” 调用失败：{e}，切换下一个…{video_path}")
            # 继续到下一个 key

        except Exception as e:
            # traceback.print_exc()
            # 未知错误直接返回，或者根据需求也可以继续尝试下一个
            return f"处理过程中发生未知错误: {e}"

        finally:
            # 无论成功还是失败，都尝试删掉已经上传的文件
            if video_file is not None:
                try:
                    print(f"[INFO] 删除临时文件 {video_file.name}…")
                    genai_flash.delete_file(video_file.name)
                except Exception as de:
                    # 如果删除也失败了，打印日志但不中断流程
                    print(f"[ERROR] 删除文件 {video_file.name} 失败：{de}")

    return f"所有 API Key 均尝试失败。最后一次错误：{last_error} {video_path}"



def get_llm_content_gemini2flash(prompt: str = '你好，Gemini！请介绍一下你自己。') -> str:
    print("[INFO] 使用模型: gemini-2.0-flash")
    last_error = None
    ordered_keys = api_key_manager.get_ordered_keys()
    for key_name in ordered_keys:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key: continue
        try:
            print(f"[INFO] 正在使用名为 '{key_name}' 的 API Key... prompt length: {len(prompt)}")
            client = genai.Client(api_key=api_key)
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            config = types.GenerateContentConfig(response_mime_type="text/plain")
            response = client.models.generate_content(model="gemini-2.0-flash", contents=contents, config=config)

            api_key_manager.record_success(key_name)  # 成功，记录
            return response.text
        except (ga_exceptions.PermissionDenied, ga_exceptions.ResourceExhausted, ga_exceptions.GoogleAPICallError) as e:
            last_error = e
            print(f"[WARN] 名为 '{key_name}' 的 API Key 调用失败: {e.__class__.__name__}. 正在尝试下一个...")
            continue
    return f"所有 API Key 均尝试失败。最后一次错误 (来自密钥 '{key_name}'): {last_error}"


def get_llm_content_sub(prompt: str = '你好，Gemini！请介绍一下你自己。',
                        model_name: str = "gemini-2.5-flash") -> str:
    print(f"[INFO] 使用模型: {model_name}")
    last_error = None
    ordered_keys = api_key_manager.get_ordered_keys()
    for key_name in ordered_keys:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key: continue
        try:
            api_key_manager.record_success(key_name)  # 成功，记录

            print(f"[INFO] 正在使用名为 '{key_name}' 的 API Key... prompt length: {len(prompt)}")
            client = genai.Client(api_key=api_key)
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=24576),
                                                 response_mime_type="text/plain")
            response = client.models.generate_content(model=model_name, contents=contents, config=config)

            return response.text
        except Exception as e:
            print(f"[WARN] 名为 '{key_name}' 的 API Key 调用失败: {e.__class__.__name__}. 正在尝试下一个... {e}")
            last_error = e
            # traceback.print_exc()
            continue
    raise last_error if last_error else Exception(f"所有 API Key 均尝试失败且未记录特定错误。")


@with_proxy
def get_llm_content(prompt: str = '你好，Gemini！请介绍一下你自己。', model_name: str = "gemini-2.5-pro") -> str | None:
    try:
        try:
            return get_llm_content_sub(prompt, model_name)
        except Exception as e1:
            print(f"[WARN] 主模型失败: {e1}")
            try:
                return get_llm_content_sub(prompt, "gemini-2.5-flash")
            except Exception as e2:
                print(f"[WARN] 备用模型失败: {e2}")
                return get_llm_content_gemini2flash(prompt)
    except Exception as e:
        print(f"[ERROR] 内容生成失败: {e}")
        print("[TIPS] 请检查以下内容：\n - API 密钥是否正确\n - 网络连接及代理设置\n - 是否安装了 `google-genai`")
        return None


@with_proxy
def analyze_images_gemini(prompt='每张图片的内容是什么', image_paths=['a.jpg']) -> str:
    last_error = None
    ordered_keys = api_key_manager.get_ordered_keys()
    for key_name in ordered_keys:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key: continue
        try:
            api_key_manager.record_success(key_name)  # 成功，记录

            print(f"[INFO] 正在使用名为 '{key_name}' 的 API Key 尝试分析图片... prompt length: {len(prompt)}, 图片数量: {len(image_paths)}")
            genai_flash.configure(api_key=api_key)
            prompt_parts = [prompt, "下面我将以'文件名:'的格式，在每个图片前提供其名称，请据此作答。"]
            for path in image_paths:
                if not os.path.exists(path): return f"错误: 图片文件未找到 -> {path}"
                prompt_parts.append(f"{os.path.basename(path)}:")
                prompt_parts.append(Image.open(path))

            model = genai_flash.GenerativeModel(model_name="gemini-2.5-pro")
            response = model.generate_content(prompt_parts, request_options={"timeout": 600})

            return response.text
        except (ga_exceptions.PermissionDenied, ga_exceptions.ResourceExhausted, ga_exceptions.GoogleAPICallError) as e:
            last_error = e
            print(f"[WARN] 名为 '{key_name}' 的 API Key 调用失败: {e.__class__.__name__}. 正在尝试下一个...")
            continue
        except Exception as e:
            return f"处理过程中发生未知错误: {e.__class__.__name__}: {e}"
    return f"所有 API Key 均尝试失败。最后一次错误 (来自密钥 '{key_name}'): {last_error}"


if __name__ == "__main__":
    print("\n" + "=" * 20 + " 开始测试 " + "=" * 20)

    print("[TEST] 正在测试 get_llm_content (这将触发第一次动态排序)")
    start_time = time.time()
    result = get_llm_content(prompt="再给我讲个笑话吧", model_name="gemini-2.5-flash")
    if result:
        print("\n[RESULT] 模型输出：\n", result)
    else:
        print(f"\n[FAIL] 内容生成失败{result}")
    print(f"[INFO] 执行时间: {time.time() - start_time:.2f} 秒")
    #
    # # 再次调用，观察排序是否根据上次成功结果发生变化
    # print("\n[TEST] 再次测试 get_llm_content (观察密钥顺序是否变化)")
    # start_time = time.time()
    # result = get_llm_content(prompt="再给我讲个笑话吧", model_name="gemini-2.5-flash")
    # if result:
    #     print("\n[RESULT] 模型输出：\n", result)
    # else:
    #     print(f"\n[FAIL] 内容生成失败{result}")
    # print(f"[INFO] 执行时间: {time.time() - start_time:.2f} 秒")
    #
    # print("\n" + "=" * 20 + " 测试结束 " + "=" * 20)