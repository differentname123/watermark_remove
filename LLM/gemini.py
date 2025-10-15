import os
import time
import base64
import traceback
from pathlib import Path

# 假设 common_utils 是您本地的模块
from common_utils.common_utils import get_config, read_json
from PIL import Image
import functools
import json
import threading
from filelock import FileLock  # 新增：导入文件锁库

# 新增：用于识别图片文件类型
import mimetypes
import uuid
from io import BytesIO

# 导入新版 SDK（必须先 pip install google-genai）
from google import genai
from google.genai import types
# 兼容旧的 google.api_core 异常导入（你的代码中已有引用）
from google.api_core import exceptions as ga_exceptions

# === 修改开始：引入健壮的、支持并发的 ApiKeyManager ===
def build_api_key_map():
    google_config = read_json(str(Path(__file__).resolve().parent / 'config_google.json'))

    detail_info_list = google_config.get('detail_info_list', [])
    api_key_map = {}
    for detail_info in detail_info_list:
        nick_name = detail_info.get('nick_name')
        gemini_api_key_list = detail_info.get('gemini_api_key_list', [])
        for index, api_key_info in enumerate(gemini_api_key_list):
            key = f'{nick_name}_{index}' if index > 0 else nick_name
            api_key_map[key] = api_key_info['api_key']

    return api_key_map

class ApiKeyManager:
    """
    一个线程安全和进程安全的API密钥管理器。
    - 每次请求时根据特定模型的使用次数动态排序密钥。
    - 使用文件锁来处理并发读写。
    """

    def __init__(self, api_key_map):
        self.api_key_map = api_key_map

        # --- 这是唯一的、核心的修改 ---
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.stats_file = os.path.join(module_dir, 'api_key_usage.json')
        # --- 修改结束 ---

        self.lock_file = self.stats_file + '.lock'
        self.lock = FileLock(self.lock_file, timeout=10)
        self._initialize_stats()

    def _initialize_stats(self):
        """初始化统计文件，如果不存在或为空。"""
        with self.lock:
            if not os.path.exists(self.stats_file) or os.path.getsize(self.stats_file) == 0:
                initial_stats = {key: {} for key in self.api_key_map.keys()}
                with open(self.stats_file, 'w') as f:
                    json.dump(initial_stats, f, indent=4)

    def get_ordered_keys(self, model_name: str):
        """
        【核心】获取根据【特定模型】使用次数动态排序的密钥名称列表。
        此操作是线程和进程安全的。
        """
        with self.lock:
            try:
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
                if stats and isinstance(next(iter(stats.values()), None), int):
                   raise TypeError("Old stats format detected. Resetting.")
            except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
                print(f"[WARN] Failed to read or parse stats file ({e}), re-initializing.")
                stats = {key: {} for key in self.api_key_map.keys()}

            for key in self.api_key_map.keys():
                if key not in stats or not isinstance(stats[key], dict):
                    stats[key] = {}

        sorted_keys = sorted(stats.keys(), key=lambda k: stats.get(k, {}).get(model_name, 0))
        print(f"[INFO] 针对模型 '{model_name}'，API 密钥将按以下动态顺序尝试: {sorted_keys}")
        return sorted_keys

    def record_success(self, key_name: str, model_name: str):
        """
        【核心】为一个成功的API调用记录指定模型的次数。
        此操作是线程和进程安全的“读取-修改-写入”原子操作。
        """
        with self.lock:
            try:
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
                if stats and isinstance(next(iter(stats.values()), None), int):
                    raise TypeError("Old stats format detected. Resetting.")
            except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
                print(f"[WARN] Failed to read or parse stats file ({e}), re-initializing.")
                stats = {key: {} for key in self.api_key_map.keys()}

            if key_name not in stats or not isinstance(stats.get(key_name), dict):
                stats[key_name] = {}

            stats[key_name][model_name] = stats[key_name].get(model_name, 0) + 1

            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            print(f"[INFO] 密钥 '{key_name}' 模型 '{model_name}' 使用次数已更新为: {stats[key_name][model_name]} 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


API_KEY_MAP = build_api_key_map()
api_key_manager = ApiKeyManager(API_KEY_MAP)
# === 修改结束 ===


# ================== 新增兼容层：在不改动现有逻辑的前提下，用新版 SDK 实现旧接口 ==================
class GenAIFlashCompat:
    """
    提供与旧 google.generativeai 模块类似的接口：
    - configure(api_key=...)
    - upload_file(path=...)
    - get_file(name)
    - delete_file(name)
    - GenerativeModel(model_name=...) -> object with generate_content(contents, request_options=...)
    以及在内部把 PIL.Image -> 上传文件 的转换（保留原始逻辑）
    """
    def __init__(self):
        self._client = None

    def configure(self, api_key: str):
        # 保存 client 实例，后续 API 调用使用该 client
        self._client = genai.Client(api_key=api_key)

    def _ensure_client(self):
        if self._client is None:
            raise RuntimeError("GenAIFlashCompat: client not configured. Call configure(api_key=...) first.")

    def upload_file(self, path: str):
        """
        path: 本地文件路径
        返回：新版 SDK 的 File 对象（保持属性 .name, .state.name 可用）
        """
        self._ensure_client()
        # 新 SDK 的接口：client.files.upload(file=path)
        return self._client.files.upload(file=path)

    def get_file(self, name: str):
        self._ensure_client()
        return self._client.files.get(name=name)

    def delete_file(self, name: str):
        self._ensure_client()
        return self._client.files.delete(name=name)

    class GenerativeModel:
        def __init__(self, outer, model_name: str):
            """
            outer: GenAIFlashCompat 的实例，用于访问 client
            """
            self._outer = outer
            self.model_name = model_name

        def generate_content(self, contents, request_options: dict | None = None, model_name: str | None = None):
            """
            contents: 原始内容列表（可能包含上传后的 File 对象、字符串、PIL.Image 等）
            request_options: 兼容旧参数（主要提取 timeout）
            model_name: 可选，调用时覆盖模型名；若为 None，则使用实例的 model_name；若实例也没有，则使用默认 'gemini-flash-latest'
            """
            self._outer._ensure_client()
            client = self._outer._client

            # 决定最终使用的模型名（优先级：调用参数 > 实例属性 > 默认）
            model_to_use = model_name or getattr(self, "model_name", None) or "gemini-flash-latest"

            # 处理 contents：把 PIL.Image -> 临时上传文件
            temp_uploaded_files = []  # 记录需要在调用后删除的临时 uploaded file names（仅临时产生的）
            processed_contents = []
            try:
                for c in contents:
                    # 如果是 PIL 图像实例，保存到临时文件并上传
                    if isinstance(c, Image.Image):
                        tmp_name = f"tmp_upload_{uuid.uuid4().hex}.png"
                        tmp_path = os.path.join(os.path.dirname(__file__), tmp_name)
                        c.save(tmp_path, format="PNG")
                        uploaded = client.files.upload(file=tmp_path)
                        processed_contents.append(uploaded)
                        temp_uploaded_files.append((uploaded.name, tmp_path))
                    else:
                        # 直接传递（例如已有上传的 File 对象、字符串等）
                        processed_contents.append(c)

                # 从 request_options 中提取 timeout（如果有）
                timeout = None
                if request_options and isinstance(request_options, dict):
                    timeout = request_options.get("timeout", None)

                # 调用新版 SDK：优先把 timeout 以关键字传入（多数版本支持）
                try:
                    if timeout is not None:
                        resp = client.models.generate_content(model=model_to_use, contents=processed_contents, timeout=timeout)
                    else:
                        resp = client.models.generate_content(model=model_to_use, contents=processed_contents)
                except TypeError:
                    # 若某些 SDK 版本不接受 timeout 参数，则降级到不带 timeout 的调用
                    resp = client.models.generate_content(model=model_to_use, contents=processed_contents)

                return resp
            finally:
                # 清理临时本地文件（如果生成了）以及删除临时上传的文件（如果已上传）
                for uploaded_name, tmp_path in temp_uploaded_files:
                    try:
                        client.files.delete(name=uploaded_name)
                    except Exception:
                        pass
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass


# 实例化兼容对象，替代原先直接 import 的 genai_flash 模块
genai_flash = GenAIFlashCompat()
# =======================================================================================


# 下面保留你原有的 with_proxy 装饰器和函数逻辑（不改动逻辑，仅调用兼容层）
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
    timeout: int = 1200
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
    # 指定要使用的模型
    model_name = "gemini-2.5-pro"
    ordered_keys = api_key_manager.get_ordered_keys(model_name=model_name)

    for key_name in ordered_keys:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key:
            continue

        # 使用新版客户端通过兼容层进行配置
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

            # 3. 调用 Gemini-2.5-Pro 生成内容（通过兼容层）
            print(f"[INFO] 使用 API Key “{key_name}” 调用 Gemini 模型 {model_name} 生成内容…")
            model = genai_flash.GenerativeModel(genai_flash, model_name=model_name)
            response = model.generate_content(
                prompt_parts,
                request_options={"timeout": timeout}
            )

            # 4. 记录成功并清理上传文件
            api_key_manager.record_success(key_name, model_name=model_name)
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
    model_name: str = "gemini-flash-latest",
    max_attempts: int = 1  # <--- 修改点 1: 增加一个控制尝试次数的参数，默认为 1
) -> str:
    last_error = None
    ordered_keys = api_key_manager.get_ordered_keys(model_name=model_name)

    # <--- 修改点 2: 使用列表切片来限制循环次数
    for key_name in ordered_keys[:max_attempts]:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key:
            continue

        # 使用新版 SDK（兼容层）
        genai_flash.configure(api_key=api_key)
        if not os.path.exists(video_path):
            return f"错误: 视频文件未找到 -> {video_path}"

        video_file = None
        try:
            print(f"[INFO] 使用 API Key “{key_name}” prompt length: {len(prompt)} 上传视频… {model_name}， {video_path}")
            api_key_manager.record_success(key_name, model_name=model_name)
            video_file = genai_flash.upload_file(path=video_path)

            while video_file.state.name == "PROCESSING":
                print("等待视频处理完成…")
                time.sleep(10)
                video_file = genai_flash.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise RuntimeError(f"视频处理失败：{video_path}")

            model = genai_flash.GenerativeModel(genai_flash, model_name=model_name)
            response = model.generate_content(
                [video_file, prompt],
                request_options={"timeout": 1200}
            )

            return response.text

        except (ga_exceptions.PermissionDenied,
                ga_exceptions.ResourceExhausted,
                ga_exceptions.GoogleAPICallError) as e:
            last_error = e
            print(f"[WARN] Key “{key_name}” 调用失败：{e}，切换下一个…{video_path}")
            # 继续到下一个 key

        except Exception as e:
            return f"处理过程中发生未知错误: {e}"

        finally:
            if video_file is not None:
                try:
                    print(f"[INFO] 删除临时文件 {video_file.name}…")
                    genai_flash.delete_file(video_file.name)
                except Exception as de:
                    print(f"[ERROR] 删除文件 {video_file.name} 失败：{de}")

    return f"所有 API Key 均尝试失败 ({max_attempts}次)。最后一次错误：{last_error} {video_path}"



def get_llm_content_gemini2flash(prompt: str = '你好，Gemini！请介绍一下你自己。') -> str:
    last_error = None
    model_name = "gemini-flash-latest"
    ordered_keys = api_key_manager.get_ordered_keys(model_name=model_name)
    for key_name in ordered_keys:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key: continue
        try:
            print(f"[INFO] 正在使用名为 '{key_name}' 的 API Key... prompt length: {len(prompt)}")
            client = genai.Client(api_key=api_key)
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            config = types.GenerateContentConfig(response_mime_type="text/plain")
            response = client.models.generate_content(model=model_name, contents=contents, config=config)

            api_key_manager.record_success(key_name, model_name=model_name)  # 成功，记录
            return response.text
        except (ga_exceptions.PermissionDenied, ga_exceptions.ResourceExhausted, ga_exceptions.GoogleAPICallError) as e:
            last_error = e
            print(f"[WARN] 名为 '{key_name}' 的 API Key 调用失败: {e.__class__.__name__}. 正在尝试下一个...")
            continue
    return f"所有 API Key 均尝试失败。最后一次错误 (来自密钥 '{key_name}'): {last_error}"


def get_llm_content_sub(prompt: str = '你好，Gemini！请介绍一下你自己。',
                        model_name: str = "gemini-flash-latest") -> str:
    print(f"[INFO] 使用模型: {model_name}")
    last_error = None
    ordered_keys = api_key_manager.get_ordered_keys(model_name=model_name)
    for key_name in ordered_keys:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key: continue
        try:
            api_key_manager.record_success(key_name, model_name=model_name)

            print(f"[INFO] 正在使用名为 '{key_name}' 的 API Key... prompt length: {len(prompt)}")
            client = genai.Client(api_key=api_key)
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=24576),
                                                 response_mime_type="text/plain")
            response = client.models.generate_content(model=model_name, contents=contents, config=config)
            text = response.text
            if not text:
                print(f"模型返回了空响应{response.prompt_feedback}")
            return text
        except Exception as e:
            print(f"[WARN] 名为 '{key_name}' 的 API Key 调用失败: {e.__class__.__name__}. 正在尝试下一个... {e}")
            last_error = e
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
                return get_llm_content_sub(prompt, "gemini-flash-latest")
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
    model_name = "gemini-2.5-pro"
    ordered_keys = api_key_manager.get_ordered_keys(model_name=model_name)
    for key_name in ordered_keys:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key: continue
        try:
            api_key_manager.record_success(key_name, model_name=model_name)

            print(f"[INFO] 正在使用名为 '{key_name}' 的 API Key 尝试分析图片... prompt length: {len(prompt)}, 图片数量: {len(image_paths)}")
            genai_flash.configure(api_key=api_key)
            prompt_parts = [prompt, "下面我将以'文件名:'的格式，在每个图片前提供其名称，请据此作答。"]
            for path in image_paths:
                if not os.path.exists(path): return f"错误: 图片文件未找到 -> {path}"
                prompt_parts.append(f"{os.path.basename(path)}:")
                # 保留你的原始逻辑：向 prompt_parts 中放入 PIL.Image 对象（兼容层会在发送前上传）
                prompt_parts.append(Image.open(path))

            model = genai_flash.GenerativeModel(genai_flash, model_name=model_name)
            response = model.generate_content(prompt_parts, request_options={"timeout": 1200})

            return response.text
        except (ga_exceptions.PermissionDenied, ga_exceptions.ResourceExhausted, ga_exceptions.GoogleAPICallError) as e:
            last_error = e
            print(f"[WARN] 名为 '{key_name}' 的 API Key 调用失败: {e.__class__.__name__}. 正在尝试下一个...")
            continue
        except Exception as e:
            return f"处理过程中发生未知错误: {e.__class__.__name__}: {e}"
    return f"所有 API Key 均尝试失败。最后一次错误 (来自密钥 '{key_name}'): {last_error}"


@with_proxy
def valid_all_api_keys():
    """
    测试所有 API Key 的有效性。
    """
    failed_key_list = []
    success_key_list = []
    # 测试时，我们基于最常用或基础的模型（如flash）来排序
    test_model = "gemini-flash-latest"
    ordered_keys = api_key_manager.get_ordered_keys(model_name=test_model)
    results = {}
    for key_name in ordered_keys:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key:
            results[key_name] = "无效（未配置）"
            continue
        try:
            print(f"[TEST] 正在测试名为 '{key_name}' 的 API Key...")
            client = genai.Client(api_key=api_key)
            contents = [types.Content(role="user", parts=[types.Part.from_text(text="你好")])]
            config = types.GenerateContentConfig(response_mime_type="text/plain")
            response = client.models.generate_content(model=test_model, contents=contents, config=config)
            results[key_name] = "有效"
            print(f"[SUCCESS] Key '{key_name}' 有效，模型响应: {response.text[:30]}...")
            success_key_list.append(key_name)
        except Exception as e:
            results[key_name] = f"无效 {api_key}（{e.__class__.__name__}: {e})"
            print(f"[FAIL] Key '{key_name}' 无效: {e}")
            failed_key_list.append(key_name)
    print("\n=== API Key 测试结果 ===")
    for k, v in results.items():
        print(f"- {k}: {v}")

    print(f"\n总计: {len(ordered_keys)} 个 Key, 成功: {len(success_key_list)}, 失败: {len(failed_key_list)}")
    print("失败的 Key 列表:", failed_key_list)
    print("成功的 Key 列表:", success_key_list)


if __name__ == "__main__":

    valid_all_api_keys()

    print("\n" + "=" * 20 + " 开始测试 " + "=" * 20)
    print("[TEST] 正在测试 get_llm_content (这将触发第一次动态排序)")
    start_time = time.time()
    # model_name 参数现在会影响密钥的选择顺序
    result = get_llm_content(prompt="再给我讲个笑话吧", model_name="gemini-flash-latest")
    if result:
        print("\n[RESULT] 模型输出：\n", result)
    else:
        print(f"\n[FAIL] 内容生成失败{result}")
    print(f"[INFO] 执行时间: {time.time() - start_time:.2f} 秒")
