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
    通过原子性的“检出”操作，实现线程/进程安全的 API Key 负载均衡。
    """

    def __init__(self, api_key_map):
        self.api_key_map = api_key_map
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.stats_file = os.path.join(module_dir, 'api_key_usage.json')
        self.lock_file = self.stats_file + '.lock'
        # 增加超时以防高并发场景下的锁等待
        self.lock = FileLock(self.lock_file, timeout=20)
        self._initialize_stats()

    def _initialize_stats(self):
        with self.lock:
            if not os.path.exists(self.stats_file) or os.path.getsize(self.stats_file) == 0:
                initial_stats = {key: {} for key in self.api_key_map.keys()}
                with open(self.stats_file, 'w') as f:
                    json.dump(initial_stats, f, indent=4)

    def _read_stats_safely(self):
        """内部辅助函数，用于在锁内安全地读取和验证统计数据。"""
        try:
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            # 兼容旧格式或修复损坏的数据
            if stats and isinstance(next(iter(stats.values()), None), int):
                raise TypeError("Old stats format detected. Resetting.")

            # 确保所有当前的key都存在于统计文件中
            for key in self.api_key_map.keys():
                if key not in stats or not isinstance(stats[key], dict):
                    stats[key] = {}
            return stats
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            print(f"[WARN] 无法读取或解析统计文件 ({e})，正在重新初始化。")
            return {key: {} for key in self.api_key_map.keys()}

    def checkout_key(self, model_name: str) -> str | None:
        """
        原子性地获取并标记一个使用次数最少的 Key。这是解决并发问题的核心。
        """
        with self.lock:
            stats = self._read_stats_safely()

            # 1. 找到使用次数最少的 key (仅在当前配置的 api_key_map 中寻找)
            valid_keys = [k for k in self.api_key_map.keys()]
            if not valid_keys:
                return None

            selected_key = min(valid_keys, key=lambda k: stats.get(k, {}).get(model_name, 0))

            # 2. 立即增加其使用次数
            stats[selected_key][model_name] = stats.get(selected_key, {}).get(model_name, 0) + 1

            # 3. 写回文件
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=4)

            print(
                f"[INFO] 检出 Key: '{selected_key}' 用于模型 '{model_name}'。新计数: {stats[selected_key][model_name]}. 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            # 4. 返回被选中的 key
            return selected_key

    def get_ordered_keys(self, model_name: str):
        """
        获取一个基于当前使用计数的有序列表。
        注意：此方法本身不是为并发选择key而设计的，主要用于非并发场景或调试。
        """
        with self.lock:
            stats = self._read_stats_safely()

        # 仅对存在于当前配置中的 key 进行排序
        sorted_keys = sorted(
            [k for k in self.api_key_map.keys()],
            key=lambda k: stats.get(k, {}).get(model_name, 0)
        )
        print(f"[INFO] 针对模型 '{model_name}'，API 密钥当前的使用顺序 (仅供参考): {sorted_keys}")
        return sorted_keys

    def record_success(self, key_name: str, model_name: str):
        """
        在新的 checkout 模式下，此方法是多余的，因为计数已在检出时增加。
        保留此方法以兼容旧代码，但它不执行任何操作。
        """
        pass  # 在 checkout 模式下，计数在检出时完成。


# ==================== 新增：违禁视频管理器 ====================
class ProhibitedVideoManager:
    """
    通过文件锁，实现一个进程安全的、持久化的违禁视频路径列表。
    """

    def __init__(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.list_file = os.path.join(module_dir, 'prohibited_videos.json')
        self.lock_file = self.list_file + '.lock'
        self.lock = FileLock(self.lock_file, timeout=20)
        self._initialize_list()

    def _initialize_list(self):
        """如果列表文件不存在，则创建一个包含空列表的初始文件。"""
        with self.lock:
            if not os.path.exists(self.list_file) or os.path.getsize(self.list_file) == 0:
                with open(self.list_file, 'w') as f:
                    json.dump([], f)

    def _read_list_safely(self) -> list:
        """在锁内安全地读取视频列表，处理文件不存在或格式错误的情况。"""
        try:
            with open(self.list_file, 'r') as f:
                data = json.load(f)
                # 确保读取到的是一个列表
                return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件损坏或不存在，返回空列表
            return []

    def add_video(self, video_path: str):
        """原子性地将一个视频路径添加到违禁列表中。"""
        with self.lock:
            video_list = self._read_list_safely()
            # 确保不重复添加
            if video_path not in video_list:
                video_list.append(video_path)
                with open(self.list_file, 'w') as f:
                    json.dump(video_list, f, indent=4)
                print(f"[INFO] 已将违禁视频 '{video_path}' 添加到记录中。")

    def is_prohibited(self, video_path: str) -> bool:
        """原子性地检查一个视频路径是否在违禁列表中。"""
        with self.lock:
            video_list = self._read_list_safely()
            return video_path in video_list


API_KEY_MAP = build_api_key_map()
api_key_manager = ApiKeyManager(API_KEY_MAP)
# 新增：实例化违禁视频管理器，使其在整个应用中可用
prohibited_video_manager = ProhibitedVideoManager()


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


def safe_generate_content(client: genai.Client, model: str, contents, config: types.GenerateContentConfig,
                          timeout: int | None = None):
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


# ========== 业务函数（修改核心视频处理逻辑） ==========

@with_proxy
def get_llm_content_gemini_flash_video(
        prompt: str = '视频中的内容是什么',
        video_path: str = 'test.mp4',
        model_name: str = "gemini-flash-latest",
        max_attempts: int = 10
) -> str:
    # ==================== 新增：前置检查 ====================
    # 在处理前，先检查该视频是否已在违禁列表中
    if prohibited_video_manager.is_prohibited(video_path):
        error_msg = f"视频处理被跳过：'{video_path}' 已被标记为包含禁止内容。"
        print(f"[ERROR] {error_msg}")
        # 直接抛出异常，避免浪费资源
        raise ValueError(error_msg)
    # ======================================================

    last_error = None

    # 尝试次数不能超过可用 key 的数量
    num_available_keys = len(API_KEY_MAP)
    attempts = min(max_attempts, num_available_keys)

    for attempt in range(attempts):
        key_name = api_key_manager.checkout_key(model_name=model_name)
        if not key_name:
            print("[ERROR] 无法检出任何 API Key，停止尝试。")
            break

        api_key = API_KEY_MAP.get(key_name)
        # 这个检查理论上多余，因为 checkout_key 基于 API_KEY_MAP，但作为安全措施保留
        if not api_key:
            continue

        client = genai.Client(api_key=api_key)
        if not os.path.exists(video_path):
            return f"错误: 视频文件未找到 -> {video_path}"

        video_file = None
        try:
            # 日志更新以反映新的尝试逻辑
            print(
                f"[INFO] 第 {attempt + 1}/{attempts} 次尝试。使用 Key “{key_name}” prompt length: {len(prompt)} 上传视频… {model_name}， {video_path}")
            # 不再需要手动调用 record_success

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
                # 将 response.prompt_feedback 转换为字符串进行通用检查
                feedback_str = str(response.prompt_feedback)
                if 'PROHIBITED_CONTENT' in feedback_str:
                    # 现在只要反馈信息中包含关键字，就能触发
                    print(f"[PROHIBITED_CONTENT] 检测到禁止内容于视频: {video_path}。正在记录并停止尝试。")
                    prohibited_video_manager.add_video(video_path)
                    # 抛出一个明确的异常，通知上层调用者这是一个不可恢复的错误
                    raise ValueError(f"PROHIBITED_CONTENT '{video_path}' contains prohibited content.")

                # 对于其他原因导致的空响应，保持原有逻辑，并确保返回字符串
                print(f"[WARN] 模型返回了空响应{feedback_str} {video_path}")
                return feedback_str
            # 成功则直接返回
            return response.text
        except Exception as e:
            if 'overloaded' in str(e) or 'An internal error has occurred' in str(e):
                last_error = e
                print(f"[WARN] Key “{key_name}”  {model_name} {prompt[:50]} {model_name} {prompt[:50]} 调用失败：{e}，切换下一个…{video_path}")
                time.sleep(600)

                # 继续循环以检出下一个key
            else:
                print(f"[ERROR] Key “{key_name}” {model_name} {prompt[:50]} 调用失败：{e}，停止尝试。 {video_path}")
                raise e  # 对于不可恢复的错误，直接抛出
        finally:
            if video_file is not None:
                try:
                    print(f"[INFO] 删除临时文件 {video_file.name}…")
                    client.files.delete(name=video_file.name)
                except Exception as de:
                    print(f"[ERROR] 删除文件 {video_file.name} 失败：{de}")

    return f"所有 API Key 均尝试失败 ({attempts}次)。最后一次错误：{last_error} {video_path}"


def get_llm_content_gemini2flash(prompt: str = '你好，Gemini！请介绍一下你自己。') -> str:
    last_error = None
    model_name = "gemini-flash-latest"
    last_key_name = None

    # 尝试所有可用的key
    num_available_keys = len(API_KEY_MAP)
    for attempt in range(num_available_keys):
        key_name = api_key_manager.checkout_key(model_name=model_name)
        if not key_name:
            print("[ERROR] 无法检出任何 API Key，停止尝试。")
            break

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

            # 不再需要手动调用 record_success
            return response.text
        except Exception as e:
            if 'overloaded' in str(e) or 'An internal error has occurred' in str(e):
                last_error = e
                print(f"[WARN] 名为 '{key_name}' 的 API Key {model_name} {prompt[:50]} 调用失败: {e.__class__.__name__}. 正在尝试下一个...")
                time.sleep(600)

                continue
            else:
                print(f"[ERROR] 名为 '{key_name}' 的 API Key {model_name} {prompt[:50]} 调用失败: {e.__class__.__name__}. 停止尝试。 {e}")
                raise e
    return f"所有 API Key 均尝试失败。最后一次错误 (来自密钥 '{last_key_name}')：{last_error}"


def get_llm_content_sub(prompt: str = '你好，Gemini！请介绍一下你自己。',
                        model_name: str = "gemini-flash-latest") -> str:
    print(f"[INFO] 使用模型: {model_name}")
    last_error = None

    # 尝试所有可用的key
    num_available_keys = len(API_KEY_MAP)
    for attempt in range(num_available_keys):
        key_name = api_key_manager.checkout_key(model_name=model_name)
        if not key_name:
            print("[ERROR] 无法检出任何 API Key，停止尝试。")
            break

        api_key = API_KEY_MAP.get(key_name)
        if not api_key:
            continue
        try:
            # 不再需要手动调用 record_success
            print(f"[INFO] 正在使用名为 '{key_name}' 的 API Key... prompt length: {len(prompt)}")
            client = genai.Client(api_key=api_key)
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            config = build_generate_content_config(model_name)
            response = safe_generate_content(client, model_name, contents, config, timeout=None)

            text = response.text
            if not text:
                print(f"模型返回了空响应{response.prompt_feedback}")
                return response.prompt_feedback
            return text
        except Exception as e:
            if 'overloaded' in str(e) or 'An internal error has occurred' in str(e):
                print(f"[WARN] 名为 '{key_name}' 的 API Key {model_name} {prompt[:50]} 调用失败: {e.__class__.__name__}. 正在尝试下一个... {e}")
                last_error = e
                time.sleep(600)

                continue
            else:
                print(f"[ERROR] 名为 '{key_name}' 的 API Key {model_name} {prompt[:50]} 调用失败: {e.__class__.__name__}. 停止尝试。 {e}")
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
    此函数按顺序测试，不涉及并发，因此使用 get_ordered_keys 是合适的。
    """
    failed_key_list = []
    success_key_list = []
    test_model = "gemini-flash-lite-latest"
    # 这里使用 get_ordered_keys 保持原样，以便按使用频率顺序测试
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