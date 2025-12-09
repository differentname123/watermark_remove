import functools
import os
import subprocess
import json
import traceback
import time
from filelock import FileLock, Timeout

# --- 新增部分 ---
# 1. 定义锁文件的数量和基础名称。
MAX_CONCURRENT_TASKS = 10
LOCK_FILE_TEMPLATE = os.path.join(os.path.dirname(__file__), "gemini.process.lock.{}")
# --- 新增部分结束 ---

def with_proxy(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
        os.environ['no_proxy'] = 'localhost,127.0.0.1'

        try:
            return func(*args, **kwargs)
        finally:
            if 'HTTP_PROXY' in os.environ:
                del os.environ['HTTP_PROXY']
            if 'HTTPS_PROXY' in os.environ:
                del os.environ['HTTPS_PROXY']

    return wrapper

@with_proxy
def ask_gemini(prompt, model_name='gemini-2.5-flash'):
    """
    通过调用 gemini-cli 向 Gemini 提问并返回文本结果。

    【高并发控制版 - 跨平台】:
    此版本使用文件锁来模拟信号量，确保在任何操作系统上，
    实际执行核心代码的进程数最多为 2。
    """
    my_lock = None

    # --- 并发控制部分 ---
    print(f"[进程 {os.getpid()}] 正在尝试获取文件锁许可...")
    while my_lock is None:
        for i in range(MAX_CONCURRENT_TASKS):
            try:
                lock_path = LOCK_FILE_TEMPLATE.format(i)
                lock = FileLock(lock_path)
                lock.acquire(timeout=0)
                my_lock = lock
                print(f"[进程 {os.getpid()}] 已获得许可 (lock {i})，开始执行 gemini-cli。")
                break
            except Timeout:
                continue
        if my_lock is None:
            time.sleep(0.5)

    # --- 使用 try...finally 确保锁一定会被释放 ---
    try:
        # vvvvvv 这里是函数核心逻辑 vvvvvv
        try:
            npm_path = os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'npm')
            gemini_executable = os.path.join(npm_path, 'gemini.cmd' if os.name == 'nt' else 'gemini')
            command = [gemini_executable, '-m', model_name, '-o', 'json']

            result = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                check=True,  # check=True 会在返回码非0时抛出 CalledProcessError
                encoding='utf-8'
            )

            response_data = json.loads(result.stdout)
            text_content = response_data.get('response')
            return text_content.strip()

        except subprocess.CalledProcessError as e:
            # 当 gemini-cli 调用失败时，捕获异常并打印详细信息
            print("=" * 20 + " gemini-cli 调用失败，打印详细信息 " + "=" * 20)
            print(f"命令: {' '.join(e.cmd)}")
            print(f"返回码 (Exit Code): {e.returncode}")

            print("\n--- STDOUT ---")
            print(e.stdout.strip() if e.stdout else "无标准输出。")

            print("\n--- STDERR (详细返回结果) ---")
            print(e.stderr.strip() if e.stderr else "无标准错误输出。")
            print("=" * 70)

            # 重新抛出异常，以便外部的重试逻辑能够正常工作
            raise e
        # ^^^^^^ 核心逻辑结束 ^^^^^^

    finally:
        # 无论函数成功还是异常退出，都必须释放锁
        if my_lock:
            my_lock.release()
            print(f"[进程 {os.getpid()}] 执行完毕，已释放许可 ({my_lock.lock_file})。")


# --- 业务代码 (同样无需修改) ---
if __name__ == "__main__":
    from multiprocessing import Pool

    prompts = [f"任务 {i}" for i in range(1, 7)]
    with Pool(processes=4) as pool:
        results = pool.map(ask_gemini, prompts)
    print("\n--- 所有任务完成 ---")