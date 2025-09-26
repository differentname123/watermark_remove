#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_replace_and_run.py

行为（固定）：
 1) 读取当前目录下 follow_back.py
 2) 找到第一行以 "total_cookie = " 开头的那一行（忽略缩进）
 3) 为 COOKIE_NAMES 中的每个名字，生成一个新文件 follow_back{safe_name}.py，
    将那一行替换成：
        total_cookie = get_config("<COOKIE_NAME>")
    （保留原缩进）
 4) 并发启动所有生成的文件（使用当前 Python 解释器），实时打印每个进程的 stdout/stderr（带前缀），
    并把输出写入对应的 log 文件。
 5) 主进程一直等待，直到所有子进程结束（可以用 Ctrl+C 中断，脚本会尝试终止子进程）

注意：
 - 生成并运行的脚本会真实执行 follow_back 的逻辑（可能做网络请求、修改文件等），请确保在合适的环境运行。
 - 如果 follow_back.py 使用项目相对导入或依赖包结构，直接运行生成的文件可能抛 ImportError —— 那种情况需按项目结构调整运行方式（可在生成文件顶部修改 sys.path）。
"""

import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List

# ========== 在这里写死你要的 get_config 参数（按需修改） ==========
COOKIE_NAMES = [
]
user_name_list = ['tao', 'xiaoxue', 'jie', 'qiqi', 'mama', 'xiaosu', 'jun', 'jj', 'ning', 'qiqixiao', 'ruruxiao']
for name in user_name_list:
    cookie_var = f"{name}_bilibili_total_cookie"
    if cookie_var not in COOKIE_NAMES:
        COOKIE_NAMES.append(cookie_var)


# ================================================================

SRC_FILENAME = "follow_back.py"
BACKUP_FILENAME = SRC_FILENAME + ".bak"

def safe_name(name: str) -> str:
    """生成文件安全的名字：仅保留字母数字，下划线替换其它字符"""
    return name.split('_')[0]

def replace_first_total_cookie_line(src_text: str, cookie_name: str) -> str:
    """
    找到第一处以 total_cookie = 开头的整行（含缩进），并替换为:
        <indent>total_cookie = get_config("<cookie_name>")
    如果没找到，抛异常。
    """
    pattern = re.compile(r'^[ \t]*(total_cookie\s*=.*)$', flags=re.MULTILINE)
    m = pattern.search(src_text)
    if not m:
        raise RuntimeError("源文件中未找到以 'total_cookie = ' 开头的行。")
    original_line = m.group(0)
    indent_match = re.match(r'^[ \t]*', original_line)
    indent = indent_match.group(0) if indent_match else ""
    new_line = f'{indent}total_cookie = get_config("{cookie_name}")'
    # 只替换第一处（基于 src_text）
    return pattern.sub(new_line, src_text, count=1)

def stream_reader(pipe, prefix: str, logfile_path):
    """从管道按行读取并同时打印到控制台与写日志文件"""
    with open(logfile_path, "a", encoding="utf-8") as logf:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            text = line.rstrip('\n')
            print(f"{prefix} {text}")
            logf.write(text + "\n")
        pipe.close()

def run_all_and_wait(commands: List[List[str]], prefixes: List[str], log_dir: Path):
    procs = []
    readers = []

    try:
        # 启动所有进程
        for cmd, prefix in zip(commands, prefixes):
            stdout_log = log_dir / f"{prefix}_stdout.log"
            stderr_log = log_dir / f"{prefix}_stderr.log"
            # 使用 pipes 以便实时打印
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
                encoding="utf-8",  # 用 utf-8 解码子进程输出
                errors="replace",  # 非法字节用替代字符替换，避免 UnicodeDecodeError
            )
            procs.append((p, prefix, stdout_log, stderr_log))

            # 启动线程从 stdout/stderr 读取
            t_out = threading.Thread(target=stream_reader, args=(p.stdout, f"[{prefix}][OUT]", stdout_log), daemon=True)
            t_err = threading.Thread(target=stream_reader, args=(p.stderr, f"[{prefix}][ERR]", stderr_log), daemon=True)
            t_out.start()
            t_err.start()
            readers.extend([t_out, t_err])

            print(f"[主进程] 已启动 {prefix} (pid={p.pid})")

        # 主循环：打印状态，直到所有结束
        while True:
            alive = [ (p, prefix, p.poll()) for (p, prefix, _, _), (pobj, prefix2, _, _) in zip(procs, procs) ]
            # print status
            statuses = []
            for (p, prefix, stdout_log, stderr_log) in procs:
                rc = p.poll()
                status = "running" if rc is None else f"exited({rc})"
                statuses.append(f"{prefix}:{status}")
            print("[主进程] 进程状态: " + " | ".join(statuses))
            # 若全部结束 break
            if all(p.poll() is not None for (p, _, _, _) in procs):
                break
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n[主进程] 收到 Ctrl+C，尝试终止所有子进程...")
        for (p, prefix, _, _) in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                    print(f"[主进程] 已发送 terminate 给 {prefix} (pid={p.pid})")
                except Exception as e:
                    print(f"[主进程] 终止 {prefix} 时出错: {e}")
        # 等待短暂时间再 kill
        time.sleep(2)
        for (p, prefix, _, _) in procs:
            if p.poll() is None:
                try:
                    p.kill()
                    print(f"[主进程] 已 kill {prefix} (pid={p.pid})")
                except Exception as e:
                    print(f"[主进程] kill {prefix} 时出错: {e}")
    finally:
        # 等待所有子线程结束
        for t in readers:
            t.join(timeout=1)
        # 打印每个进程最终码
        for (p, prefix, stdout_log, stderr_log) in procs:
            rc = p.poll()
            print(f"[主进程] {prefix} 最终返回码: {rc} (stdout-> {stdout_log}, stderr-> {stderr_log})")

def main():
    src_path = Path(SRC_FILENAME)
    if not src_path.exists():
        print(f"错误：未找到源文件 {SRC_FILENAME}", file=sys.stderr)
        return

    # 备份（仅当备份不存在时）
    backup_path = src_path.parent / BACKUP_FILENAME
    if not backup_path.exists():
        shutil.copy2(src_path, backup_path)
        print(f"已备份原文件到：{backup_path}")
    else:
        print(f"备份已存在：{backup_path}（未覆盖）")

    src_text = src_path.read_text(encoding="utf-8")

    generated_files = []
    for cookie in COOKIE_NAMES:
        try:
            new_text = replace_first_total_cookie_line(src_text, cookie)
        except Exception as e:
            print(f"为 cookie '{cookie}' 生成文件失败：{e}", file=sys.stderr)
            continue
        fname = f"flw_{safe_name(cookie)}.py"
        out_path = src_path.parent / fname
        out_path.write_text(new_text, encoding="utf-8")
        generated_files.append((cookie, out_path))
        print(f"已生成：{out_path}")

    if not generated_files:
        print("没有生成任何文件，退出。")
        return

    # 为每个生成的文件构造命令
    commands = []
    prefixes = []
    for cookie, out_path in generated_files:
        cmd = [sys.executable, str(out_path)]
        commands.append(cmd)
        prefixes.append(safe_name(f"flw_{cookie}"))

    # 创建日志目录
    log_dir = Path.cwd() / "fb_run_logs"
    log_dir.mkdir(exist_ok=True)

    # print("\n[主进程] 开始并发运行所有生成的脚本（无超时，主进程会一直等待直到它们全部结束）...\n")
    # run_all_and_wait(commands, prefixes, log_dir)
    # print("\n[主进程] 全部子进程已结束。")

if __name__ == "__main__":
    main()
