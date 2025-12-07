import os
import subprocess
import json
import traceback

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


def ask_gemini(prompt):
    """
    通过调用 gemini-cli 向 Gemini 提问并返回文本结果。
    这个版本解决了所有已知问题：
    1. 使用标准输入(stdin)传递prompt，避免因prompt过长导致“文件名太长”的错误。
    2. 直接定位 gemini.cmd 的路径，避免因环境变量问题导致“命令未找到”的错误。
    3. 正确解析 gemini-cli 返回的简化版 JSON 格式。

    Args:
        prompt (str): 你要向 Gemini 提问的内容，可以是任意长度。

    Returns:
        一个字符串，包含 Gemini 的回答文本。
        如果发生任何错误，则返回 None。
    """
    try:
        # 1. 构造 gemini.cmd 的完整路径，使其不依赖系统 PATH 环境变量
        # os.path.expanduser('~') 会自动获取当前用户的主目录 (例如 C:\Users\zxh)
        npm_path = os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'npm')
        gemini_executable = os.path.join(npm_path, 'gemini.cmd')

        # 提前检查可执行文件是否存在，给出更明确的错误提示
        if not os.path.exists(gemini_executable):
            print(f"错误: 在预期的路径中未找到 gemini.cmd: {gemini_executable}")
            print("请确认 gemini-cli 是否已通过 npm 全局安装。")
            return None

        # 2. 构建命令列表，注意：prompt 本身不在这里
        command = [
            gemini_executable,
            '-m', 'gemini-2.5-pro',
            '-o', 'json'
        ]

        # 3. 执行命令，并通过 'input' 参数将长文本 prompt 安全地传递给子进程
        result = subprocess.run(
            command,
            input=prompt,  # 这是解决长文本问题的关键
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )

        # 4. 解析 gemini-cli 返回的 JSON 输出
        response_data = json.loads(result.stdout)
        text_content = response_data.get('response')

        if text_content is None:
            print("错误: 未能在JSON响应中找到 'response' 字段。")
            print(f"收到的原始JSON: {result.stdout}")
            return None

        return text_content.strip()

    # 捕获所有可能的异常，使函数在出错时不会崩溃
    except subprocess.CalledProcessError as e:
        print(f"错误: gemini-cli 执行失败，返回码: {e.returncode}")
        print(f"gemini-cli 的错误输出:\n{e.stderr}")
        traceback.print_exc()
        return None
    except json.JSONDecodeError:
        print("错误: 解析返回的 JSON 失败。")
        print(f"收到的原始输出: {result.stdout}")
        traceback.print_exc()

        return None
    except Exception as e:
        print(f"发生了一个未知错误: {e}")
        traceback.print_exc()

        return None


# --- 使用示例 ---
# 当这个文件作为主程序直接运行时，以下代码会被执行
if __name__ == "__main__":
    my_prompt = "你是谁？请用中文回答。"
    print(f"提问: {my_prompt}")

    response = ask_gemini(my_prompt)

    # 如果成功获取到回答，就打印出来
    if response:
        print("\nGemini 的回答:")
        print(response)

    print("\n" + "=" * 20 + "\n")

    # 另一个例子
    code_prompt = "用Python写一个函数，计算斐波那契数列的第n项"
    print(f"提问: {code_prompt}")

    code_response = ask_gemini(code_prompt)
    if code_response:
        print("\nGemini 的回答:")
        print(code_response)