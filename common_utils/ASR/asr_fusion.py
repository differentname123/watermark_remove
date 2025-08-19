import collections

from common_utils.common_utils import read_json, save_json


def fuse_asr_results(asr_lists: list,
                     confidence_threshold: float = 0.4,
                     merge_gap_threshold: float = 0.1):
    """
    融合多个ASR识别结果，以提高准确性。

    Args:
        asr_lists (list): 一个包含多个ASR结果列表的列表。
                          例如: [asr_result_1, asr_result_2, asr_result_3]
        confidence_threshold (float): 置信度阈值。如果一个词只由一个ASR引擎识别，
                                      且其置信度低于此阈值，则会被丢弃。
                                      这有助于过滤掉单一来源的幻觉。
        merge_gap_threshold (float): 合并相邻相同单词时允许的最大时间间隔（秒）。

    Returns:
        list: 融合后的、更准确的ASR结果列表。
    """

    # --- 步骤 1: 将所有词语片段扁平化并排序 ---
    # 将所有来源的词语放入一个列表中，并添加一个'source'字段以作区分
    all_words = []
    for i, asr_list in enumerate(asr_lists):
        for word_data in asr_list:
            # 创建副本以避免修改原始输入
            word_copy = word_data.copy()
            word_copy['source'] = i
            all_words.append(word_copy)

    # 按开始时间排序，这是后续处理的基础
    all_words.sort(key=lambda x: x['start'])

    if not all_words:
        return []

    # --- 步骤 2 & 3: 分组与决策 ---
    # 遍历排序后的列表，将时间上重叠的词语分为一组，并对每组进行决策
    fused_results = []
    current_group = [all_words[0]]
    # group_end_time 记录当前分组所覆盖的最晚时间点
    group_end_time = all_words[0]['end']

    for i in range(1, len(all_words)):
        word = all_words[i]
        # 如果当前词语的开始时间在分组时间范围内，说明有重叠，加入当前组
        if word['start'] < group_end_time:
            current_group.append(word)
            # 更新分组的最晚结束时间
            group_end_time = max(group_end_time, word['end'])
        else:
            # 当前词语与上一组无重叠，意味着上一组已结束，需要处理
            resolved_word = _resolve_group(current_group, confidence_threshold)
            if resolved_word:
                fused_results.append(resolved_word)

            # 为当前词语开始一个新组
            current_group = [word]
            group_end_time = word['end']

    # 处理最后一个分组
    resolved_word = _resolve_group(current_group, confidence_threshold)
    if resolved_word:
        fused_results.append(resolved_word)

    # --- 步骤 4: 后处理 - 合并相邻的相同词语 ---
    final_results = _merge_adjacent_words(fused_results, merge_gap_threshold)

    return final_results


def _resolve_group(group: list, confidence_threshold: float):
    """
    处理一个重叠的词语分组，选出最佳词语。
    决策逻辑：结合词语出现次数和置信度总和。
    """
    if not group:
        return None

    # 使用 defaultdict 来统计每个词语的累积置信度和出现次数
    word_scores = collections.defaultdict(float)
    word_occurrences = collections.defaultdict(list)

    for item in group:
        word = item['word']
        # 使用 get 方法以防 'probability' 字段缺失，默认值为 1.0
        prob = item.get('probability', 1.0)

        # 核心：将置信度作为权重进行累加
        word_scores[word] += prob
        word_occurrences[word].append(item)

    # 找到得分最高的词语
    if not word_scores:
        return None
    best_word = max(word_scores, key=word_scores.get)

    # 过滤掉可能是幻觉的词：如果最佳词只出现一次且置信度低
    winning_segments = word_occurrences[best_word]
    if len(winning_segments) == 1 and winning_segments[0]['probability'] < confidence_threshold:
        return None

    # --- 计算融合后的时间戳和置信度 ---
    # 开始时间：取所有识别出该词的片段中最早的开始时间
    start_time = min(s['start'] for s in winning_segments)
    # 结束时间：取所有识别出该词的片段中最晚的结束时间
    end_time = max(s['end'] for s in winning_segments)
    # 置信度：取所有识别出该词的片段的平均置信度
    avg_prob = sum(s['probability'] for s in winning_segments) / len(winning_segments)

    return {
        "start": round(start_time, 3),
        "end": round(end_time, 3),
        "word": best_word,
        "probability": round(avg_prob, 3)
    }


def _merge_adjacent_words(word_list: list, gap_threshold: float):
    """
    合并结果列表中连续且相同的词语。
    """
    if not word_list:
        return []

    merged_list = [word_list[0]]
    for i in range(1, len(word_list)):
        prev = merged_list[-1]
        curr = word_list[i]

        # 如果当前词和前一个词相同，并且它们之间的时间间隔很小
        if (prev['word'] == curr['word'] and
                curr['start'] - prev['end'] <= gap_threshold):
            # 合并：更新前一个词的结束时间和置信度
            prev['end'] = curr['end']
            # 置信度可以取平均值或最大值，这里取最大值，代表最高的信心
            prev['probability'] = max(prev['probability'], curr['probability'])
        else:
            merged_list.append(curr)

    return merged_list


# --- 示例用法 ---
if __name__ == '__main__':
    SPEAKER_LOG_FILE = 'output/segments_speech.json'
    ASR_FILES = [
        {'path': 'output/test_asr_whisper.json', 'source_name': 'whisper'},
        {'path': 'output/test_asr_whisper2.json', 'source_name': 'whisper'},

        {'path': 'output/test_asr_funasr.json', 'source_name': 'funasr'}
    ]
    OUTPUT_FILE = 'output/fused_transcript_simplified.json'
    all_asr_lists = []
    for file in ASR_FILES:
        all_asr_lists.append(read_json(file['path']))

    # 执行融合
    final_result = fuse_asr_results(all_asr_lists)

    save_json(OUTPUT_FILE, final_result)
