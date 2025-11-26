import subprocess
import re
import os


def get_audio_stats(file_path):
    """
    使用 ffmpeg 的 volumedetect 滤镜分析音频信息。
    返回一个字典，包含 max_volume 和 mean_volume。
    """
    if not os.path.exists(file_path):
        return None

    # 构建分析命令 (输出到 null，因为我们只需要读取日志里的统计信息)
    # Windows 下用 NUL，Linux/Mac 用 /dev/null，这里利用 -f null - 跨平台兼容
    command = [
        'ffmpeg',
        '-i', file_path,
        '-af', 'volumedetect',
        '-f', 'null',
        '-'
    ]

    try:
        # 运行命令并捕获 stderr (因为 ffmpeg 的统计信息输出在 stderr)
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        output = result.stderr

        # 使用正则表达式提取数值
        # 输出示例: [Parsed_volumedetect_0 @ ...] max_volume: -5.1 dB
        max_vol_match = re.search(r"max_volume:\s+([-\d.]+) dB", output)
        mean_vol_match = re.search(r"mean_volume:\s+([-\d.]+) dB", output)

        if max_vol_match and mean_vol_match:
            return {
                'max': float(max_vol_match.group(1)),
                'mean': float(mean_vol_match.group(1))
            }
        else:
            print(f"无法解析文件信息: {file_path}")
            return None

    except Exception as e:
        print(f"分析出错: {e}")
        return None


def process_and_report(input_path, output_path):
    print("=" * 50)
    print(f"1. 正在分析原文件: {input_path}")
    stats_before = get_audio_stats(input_path)

    if not stats_before:
        print("分析原文件失败，终止。")
        return

    print(f"2. 正在处理 (dynaudnorm 动态标准化)...")

    # 动态标准化命令
    process_command = [
        'ffmpeg', '-y',
        '-i', input_path,
        # f=10: 反应速度极快（10毫秒）
        # g=3:  过渡窗口极小，允许音量快速变化
        # p=0.95: 目标峰值拉得更高
        '-af', 'dynaudnorm=f=10:g=3:p=0.95',
        output_path
    ]

    subprocess.run(process_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"3. 正在分析新文件: {output_path}")
    stats_after = get_audio_stats(output_path)

    if not stats_after:
        print("分析新文件失败。")
        return

    # --- 打印对比报告 ---
    print("\n" + "=" * 50)
    print(f"{'指标':<15} | {'处理前':<12} | {'处理后':<12} | {'变化效果'}")
    print("-" * 55)

    # 1. 最大音量
    diff_max = stats_after['max'] - stats_before['max']
    print(
        f"{'最大音量 (Peak)':<12} | {stats_before['max']:>6.1f} dB   | {stats_after['max']:>6.1f} dB   | {'(变大)' if diff_max > 0 else '(变小)'} {diff_max:+.1f} dB")

    # 2. 平均音量
    diff_mean = stats_after['mean'] - stats_before['mean']
    print(
        f"{'平均音量 (Mean)':<12} | {stats_before['mean']:>6.1f} dB   | {stats_after['mean']:>6.1f} dB   | {'(变大)' if diff_mean > 0 else '(变小)'} {diff_mean:+.1f} dB")

    # 3. 动态范围 (差距) - 这是你最关心的！
    range_before = stats_before['max'] - stats_before['mean']
    range_after = stats_after['max'] - stats_after['mean']
    diff_range = range_after - range_before  # 应该是负数，表示差距缩小

    print("-" * 55)
    print(
        f"{'动态差距 (Range)':<12} | {range_before:>6.1f} dB   | {range_after:>6.1f} dB   | 缩小了 {abs(diff_range):.1f} dB")
    print("=" * 50)

    if diff_range < -2:
        print("✅ 成功：声音起伏明显变小，整体听感更均匀。")
    else:
        print("⚠️ 提示：变化不大，可能是原音频已经很均匀，或者需要调整 dynaudnorm 参数。")


# --- 使用示例 ---
if __name__ == "__main__":
    # --- 配置路径 ---
    # 源文件夹 (读取这里面的文件)
    source_dir = r"W:\project\python_project\watermark_remove\content_community\app\bgm_audioback"

    # 目标文件夹 (处理后存到这里)
    target_dir = r"W:\project\python_project\watermark_remove\content_community\app\bgm_audio"

    # --- 1. 检查目录 ---
    if not os.path.exists(source_dir):
        print(f"错误: 源文件夹不存在 -> {source_dir}")
        exit()

    # 如果目标文件夹不存在，自动创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"已创建目标文件夹 -> {target_dir}")

    # --- 2. 扫描并批量处理 ---
    # 支持的音频格式后缀
    valid_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg')

    print(f"开始扫描文件夹: {source_dir} ...\n")

    count = 0
    for filename in os.listdir(source_dir):
        # 检查后缀名 (忽略大小写)
        if filename.lower().endswith(valid_extensions):
            # 构建完整路径
            input_full_path = os.path.join(source_dir, filename)

            # 构建输出路径 (保持同名)
            # 如果你想强制转成 mp3，可以把 filename 的后缀替换掉，例如:
            # output_filename = os.path.splitext(filename)[0] + ".mp3"
            output_full_path = os.path.join(target_dir, filename)

            process_and_report(input_full_path, output_full_path)
            count += 1

    print(f"\n========================================")
    print(f"全部完成! 共处理了 {count} 个音频文件。")
    print(f"输出目录: {target_dir}")