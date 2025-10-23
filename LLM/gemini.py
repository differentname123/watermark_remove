import os
import time
import json
from pathlib import Path
import functools

# 假设 common_utils 是您本地的模块
from common_utils.common_utils import read_json

from PIL import Image  # 仅用于类型提示与可选检测，不强依赖 PIL 的上传流程
from filelock import FileLock

# 新版 SDK（pip install google-genai）
from google import genai
from google.genai import types


# ========== API Key 读取与管理 ==========

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
    线程/进程安全的 API Key 使用统计与动态排序：
    - 针对特定模型维度记录使用次数
    - 使用文件锁避免并发读写冲突
    """

    def __init__(self, api_key_map):
        self.api_key_map = api_key_map
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.stats_file = os.path.join(module_dir, 'api_key_usage.json')
        self.lock_file = self.stats_file + '.lock'
        self.lock = FileLock(self.lock_file, timeout=10)
        self._initialize_stats()

    def _initialize_stats(self):
        with self.lock:
            if not os.path.exists(self.stats_file) or os.path.getsize(self.stats_file) == 0:
                initial_stats = {key: {} for key in self.api_key_map.keys()}
                with open(self.stats_file, 'w') as f:
                    json.dump(initial_stats, f, indent=4)

    def get_ordered_keys(self, model_name: str):
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


# ========== 统一的思考预算与调用工具函数 ==========

def build_generate_content_config(model_name: str | None) -> types.GenerateContentConfig:
    """
    统一生成 GenerateContentConfig：
    - 默认 thinking_budget=24567
    - 若 model_name 包含 'pro'（不区分大小写），则为 32678
    - 统一 response_mime_type 为 'text/plain'
    """
    budget = 24567
    if model_name and ('pro' in model_name.lower()):
        budget = 32678
    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        )
    ]

    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=budget),
        response_mime_type="text/plain",
        safety_settings=safety_settings
    )

def safe_generate_content(client: genai.Client, model: str, contents, config: types.GenerateContentConfig, timeout: int | None = None):
    """
    统一模型调用，兼容部分 SDK 版本不接受 timeout 的情况。
    """
    try:
        if timeout is not None:
            return client.models.generate_content(model=model, contents=contents, config=config, timeout=timeout)
        else:
            return client.models.generate_content(model=model, contents=contents, config=config)
    except TypeError:
        # 某些版本不接受 timeout
        return client.models.generate_content(model=model, contents=contents, config=config)


def wait_until_file_ready(client: genai.Client, file_obj, poll_interval: int = 10):
    """
    轮询等待文件处理完成（PROCESSING -> ACTIVE/FAILED）。
    """
    while getattr(file_obj, "state", None) and getattr(file_obj.state, "name", None) == "PROCESSING":
        time.sleep(poll_interval)
        file_obj = client.files.get(name=file_obj.name)
    if getattr(file_obj.state, "name", None) == "FAILED":
        raise RuntimeError(f"文件处理失败：{getattr(file_obj, 'name', '未知')}")
    return file_obj


# ========== 代理装饰器（保持行为） ==========

def with_proxy(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        try:
            return func(*args, **kwargs)
        finally:
            if 'HTTP_PROXY' in os.environ:
                del os.environ['HTTP_PROXY']
            if 'HTTPS_PROXY' in os.environ:
                del os.environ['HTTPS_PROXY']
    return wrapper


# ========== 业务函数（无兼容层，保持原功能） ==========


@with_proxy
def get_llm_content_gemini_flash_video(
    prompt: str = '视频中的内容是什么',
    video_path: str = 'test.mp4',
    model_name: str = "gemini-flash-latest",
    max_attempts: int = 10
) -> str:
    last_error = None
    ordered_keys = api_key_manager.get_ordered_keys(model_name=model_name)

    for key_name in ordered_keys[:max_attempts]:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key:
            continue

        client = genai.Client(api_key=api_key)
        if not os.path.exists(video_path):
            return f"错误: 视频文件未找到 -> {video_path}"

        video_file = None
        try:
            print(f"[INFO] 使用 API Key “{key_name}” prompt length: {len(prompt)} 上传视频… {model_name}， {video_path}")
            # 保持原行为：调用前先记一次成功（影响排序）
            api_key_manager.record_success(key_name, model_name=model_name)

            video_file = client.files.upload(file=video_path)
            video_file = wait_until_file_ready(client, video_file, poll_interval=10)

            config = build_generate_content_config(model_name)
            response = safe_generate_content(
                client=client,
                model=model_name,
                contents=[video_file, prompt],
                config=config,
                timeout=1200
            )
            if not response.text:
                print(f"[WARN] 模型返回了空响应{response.prompt_feedback} {video_path}")
            return response.text
        except Exception as e:
            if 'overloaded' in str(e):
                last_error = e
                print(f"[WARN] Key “{key_name}” 调用失败：{e}，切换下一个…{video_path}")
            else:
                print(f"[ERROR] Key “{key_name}” 调用失败：{e}，停止尝试。 {video_path}")
                raise e
        finally:
            if video_file is not None:
                try:
                    print(f"[INFO] 删除临时文件 {video_file.name}…")
                    client.files.delete(name=video_file.name)
                except Exception as de:
                    print(f"[ERROR] 删除文件 {video_file.name} 失败：{de}")

    return f"所有 API Key 均尝试失败 ({max_attempts}次)。最后一次错误：{last_error} {video_path}"


def get_llm_content_gemini2flash(prompt: str = '你好，Gemini！请介绍一下你自己。') -> str:
    last_error = None
    model_name = "gemini-flash-latest"
    ordered_keys = api_key_manager.get_ordered_keys(model_name=model_name)
    last_key_name = None
    for key_name in ordered_keys:
        last_key_name = key_name
        api_key = API_KEY_MAP.get(key_name)
        if not api_key:
            continue
        try:
            print(f"[INFO] 正在使用名为 '{key_name}' 的 API Key... prompt length: {len(prompt)}")
            client = genai.Client(api_key=api_key)
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            config = build_generate_content_config(model_name)
            response = safe_generate_content(client, model_name, contents, config, timeout=None)

            api_key_manager.record_success(key_name, model_name=model_name)
            return response.text
        except Exception as e:
            if 'overloaded' in str(e):
                last_error = e
                print(f"[WARN] 名为 '{key_name}' 的 API Key 调用失败: {e.__class__.__name__}. 正在尝试下一个...")
                continue
            else:
                print(f"[ERROR] 名为 '{key_name}' 的 API Key 调用失败: {e.__class__.__name__}. 停止尝试。 {e}")
                raise e
    return f"所有 API Key 均尝试失败。最后一次错误 (来自密钥 '{last_key_name}')：{last_error}"


def get_llm_content_sub(prompt: str = '你好，Gemini！请介绍一下你自己。',
                        model_name: str = "gemini-flash-latest") -> str:
    print(f"[INFO] 使用模型: {model_name}")
    last_error = None
    ordered_keys = api_key_manager.get_ordered_keys(model_name=model_name)
    for key_name in ordered_keys:
        api_key = API_KEY_MAP.get(key_name)
        if not api_key:
            continue
        try:
            # 保持原始行为：先记成功再调用
            api_key_manager.record_success(key_name, model_name=model_name)

            print(f"[INFO] 正在使用名为 '{key_name}' 的 API Key... prompt length: {len(prompt)}")
            client = genai.Client(api_key=api_key)
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            config = build_generate_content_config(model_name)
            response = safe_generate_content(client, model_name, contents, config, timeout=None)

            text = response.text
            if not text:
                print(f"模型返回了空响应{response.prompt_feedback}")
            return text
        except Exception as e:
            if 'overloaded' in str(e):
                print(f"[WARN] 名为 '{key_name}' 的 API Key 调用失败: {e.__class__.__name__}. 正在尝试下一个... {e}")
                last_error = e
                continue
            else:
                print(f"[ERROR] 名为 '{key_name}' 的 API Key 调用失败: {e.__class__.__name__}. 停止尝试。 {e}")
                raise e
    raise last_error if last_error else Exception("所有 API Key 均尝试失败且未记录特定错误。")


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
def valid_all_api_keys():
    """
    测试所有 API Key 的有效性。
    """
    failed_key_list = []
    success_key_list = []
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
            config = build_generate_content_config(test_model)
            response = safe_generate_content(client, test_model, contents, config, timeout=None)
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
    #
    # print("\n" + "=" * 20 + " 开始测试 " + "=" * 20)
    # print("[TEST] 正在测试 get_llm_content (这将触发第一次动态排序)")
    # start_time = time.time()
    # result = get_llm_content(prompt="再给我讲个笑话吧", model_name="gemini-flash-latest")
    # if result:
    #     print("\n[RESULT] 模型输出：\n", result)
    # else:
    #     print(f"\n[FAIL] 内容生成失败{result}")
    # print(f"[INFO] 执行时间: {time.time() - start_time:.2f} 秒")