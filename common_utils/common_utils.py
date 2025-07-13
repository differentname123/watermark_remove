import ast
import copy
import json
import os
import re

import cv2
import numpy as np

import pandas as pd

def string_to_object(input_str: str):
    """
    从字符串中提取并解析出 Python 列表或字典对象，设计得更加健壮。

    该函数增强了对不规范格式的容忍度，特别适合处理来自 LLM 的输出。

    核心功能：
    1.  **智能提取**: 自动在整个字符串中定位 JSON/Python 对象的边界（从第一个 '{' 或 '[' 到最后一个 '}' 或 ']'），
        忽略前导和尾随的无关文本（例如 "当然，这是您要的JSON："）。
    2.  **兼容 Markdown**: 能够处理被 ```json ... ``` 代码块包裹的内容。
    3.  **错误修正**:
        - 自动移除常见的行内 (//) 和块级 (/* */) 注释。
        - 自动移除导致 JSON 解析失败的尾随逗号 (trailing commas)。
    4.  **双引擎解析**:
        - 首先尝试使用 `json.loads`，因为它更符合标准，速度更快。
        - 如果失败，则回退到 `ast.literal_eval`，以支持 Python 特有的字面量
          （如 `None`, `True`, `False` 以及单引号字符串）。

    如果无法找到或解析出有效的对象，则抛出 ValueError 异常。

    :param input_str: 包含列表或字典的输入字符串。
    :return: 解析后的 Python 列表或字典。
    :raises ValueError: 如果无法从字符串中找到或解析出有效的对象。
    """
    try:
        # 1. 智能提取：在字符串中寻找对象边界
        try:
            # 寻找第一个 '{' 或 '['
            start_pos = min(
                i for i in (input_str.find('{'), input_str.find('[')) if i != -1
            )
            # 寻找最后一个 '}' 或 ']'
            end_pos = max(input_str.rfind('}'), input_str.rfind(']'))

            if end_pos <= start_pos:
                raise ValueError("未找到匹配的括号/方括号。")

            # 提取出最可能包含对象的子字符串
            potential_obj_str = input_str[start_pos: end_pos + 1]

        except (ValueError, IndexError):
            raise ValueError("输入字符串中未找到疑似列表或字典的结构。")

        # 2. 错误修正：清理提取出的字符串
        # 移除 JavaScript/JSONC 风格的注释
        # 移除 // 单行注释
        potential_obj_str = re.sub(r"//.*", "", potential_obj_str)
        # 移除 /* */ 多行注释
        potential_obj_str = re.sub(r"/\*[\s\S]*?\*/", "", potential_obj_str, flags=re.MULTILINE)

        # 移除尾随逗号 (例如, [1, 2,])
        potential_obj_str = re.sub(r",\s*([}\]])", r"\1", potential_obj_str)

        # 清理字符串前后的空白字符
        cleaned_str = potential_obj_str.strip()

        # 3. 双引擎解析
        # 首先尝试使用 json.loads (更标准，通常更快)
        try:
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            # 如果 json.loads 失败，回退到 ast.literal_eval (更宽容，支持 Python 语法)
            try:
                return ast.literal_eval(cleaned_str)
            except (ValueError, SyntaxError, MemoryError) as e:
                # 如果两种方法都失败，则抛出最终的异常
                error_msg = f"无法将提取的字符串解析为列表或字典。错误: {e}"
                # 附上清理后待解析的字符串片段，便于调试
                preview = (cleaned_str[:150] + '...') if len(cleaned_str) > 150 else cleaned_str
                raise ValueError(f"{error_msg}\n尝试解析的内容 (清理后): '''{preview}'''")
    except Exception as e:
        print(f"解析字符串时发生错误: {e} {input_str}")
        return None


def get_frame_at_time_safe(video_path: str, time_str: str) -> np.ndarray | None:
    """
    从视频中提取指定时间点的帧，并在发生任何错误时安全地回退到第一帧。

    - 如果成功，返回目标时间的帧。
    - 如果时间格式错误、时间超出范围或读取目标帧失败，则返回视频的第一帧。
    - 如果视频文件无法打开或无法读取第一帧，则返回 None。

    参数:
    - video_path (str): 视频文件的路径。
    - time_str (str): "HH:MM:SS" 或 "MM:SS" 格式的时间字符串。

    返回:
    - np.ndarray: OpenCV格式的图像帧。
    - None: 仅在视频文件无法打开或损坏时返回。
    """
    # 1. 尝试打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"严重错误: 无法打开视频文件: {video_path}。无法获取任何帧。")
        return None

    # 2. 立即读取第一帧作为备用
    ret_first, first_frame = cap.read()
    if not ret_first:
        print(f"严重错误: 视频 '{video_path}' 可打开但无法读取第一帧。")
        cap.release()
        return None

    try:
        # 3. 尝试解析时间并定位目标帧（正常流程）
        try:
            parts = list(map(int, time_str.split(':')))
            if len(parts) == 3:
                h, m, s = parts
            elif len(parts) == 2:
                h, m, s = 0, parts[0], parts[1]
            else:
                raise ValueError("时间格式应为 'HH:MM:SS' 或 'MM:SS'")
            total_seconds = h * 3600 + m * 60 + s
        except ValueError as e:
            # 如果时间格式解析失败，直接触发回退
            raise ValueError(f"时间格式不正确 ({e})") from e

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            raise IOError("无法读取视频的帧率 (FPS)。")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps

        if total_seconds > video_duration:
            raise ValueError(f"指定时间 {time_str} 超出视频总时长 ({video_duration:.2f}s)")

        target_frame_index = int(total_seconds * fps)

        # 对于非常接近第一帧的情况，直接使用已读取的第一帧
        if target_frame_index == 0:
            return first_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
        ret, target_frame = cap.read()

        if not ret:
            raise IOError(f"无法在时间点 {time_str} 读取到帧")

        # 如果一切顺利，返回目标帧
        return target_frame

    except Exception as e:
        # 4. 如果try块中出现任何异常，执行回退逻辑
        print(f"处理时发生异常: {e}")
        print(">>> 已触发回退机制，将返回视频的第一帧。")
        return first_frame

    finally:
        # 5. 确保无论如何都释放视频捕获对象
        cap.release()

def select_strategies_optimized(
    strategy_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    k: int,
    strategy_id_col: str = 'index',  # 新增参数：指定包含策略ID的列名
    count_col: str = 'capital_no_leverage',  # 新增参数：指定包含计数的列名
    penalty_scaler: float = 1.0,
    use_absolute_correlation: bool = True,
):
    """
    使用贪婪算法选择一组策略，ID在指定列中，自动调整惩罚因子。

    目标是最大化总count，同时最小化策略间的相关性。

    Args:
        strategy_df (pd.DataFrame): 包含策略ID列和count列的DataFrame。
        correlation_df (pd.DataFrame): 包含策略对及其相关性的DataFrame。
                                        需要有 'Row1', 'Row2', 'Correlation' 列。
                                        'Row1', 'Row2'的值应能匹配 strategy_df 中 strategy_id_col 的值。
        k (int): 希望选出的策略数量。
        strategy_id_col (str): strategy_df 中包含策略ID的列名。默认为 'index'。
        count_col (str): strategy_df 中包含 count 的列名。默认为 'capital_no_leverage'。
        penalty_scaler (float, optional): 自动计算的惩罚因子的缩放系数。
                                         默认为 1.0。大于1增加惩罚，小于1减少惩罚。
        use_absolute_correlation (bool, optional): 是否在计算惩罚时使用绝对相关性值。
                                                  默认为 True。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - pd.DataFrame: 一个包含被选中策略行的DataFrame (来自原始 strategy_df)。
                            列和索引与原始 strategy_df 保持一致, 按选择顺序排序。
            - pd.DataFrame: 只包含选定策略之间相关性的新DataFrame。
                            列为 ['Row1', 'Row2', 'Correlation']。
    """

    # --- 1. 输入验证和数据准备 ---
    if strategy_id_col not in strategy_df.columns:
        raise ValueError(f"strategy_df 必须包含策略ID列: '{strategy_id_col}'")
    if count_col not in strategy_df.columns:
        raise ValueError(f"strategy_df 必须包含列: '{count_col}'")
    if not all(col in correlation_df.columns for col in ['Row1', 'Row2', 'Correlation']):
        raise ValueError("correlation_df 必须包含列: 'Row1', 'Row2', 'Correlation'")
    if k <= 0:
        empty_strategies = strategy_df.iloc[0:0]  # 返回与输入结构相同的空DF
        empty_correlations = pd.DataFrame(columns=['Row1', 'Row2', 'Correlation'])
        return empty_strategies, empty_correlations

    # 检查策略ID列是否有重复值，这可能导致问题
    if strategy_df[strategy_id_col].duplicated().any():
        print(f"警告: 策略ID列 '{strategy_id_col}' 中存在重复值。这可能影响结果的准确性。")

    # 复制以防修改原始df
    original_strat_df = strategy_df.copy()
    strat_df_internal = strategy_df.copy()
    strat_df_internal['_internal_id_str'] = strat_df_internal[strategy_id_col].astype(str).str.strip()
    strat_df_internal = strat_df_internal.set_index('_internal_id_str', drop=True)  # 使用临时字符串ID列作为索引

    corr_df = correlation_df.copy()

    corr_df['Row1'] = corr_df['Row1'].astype(str).str.strip()
    corr_df['Row2'] = corr_df['Row2'].astype(str).str.strip()

    # --- 自动计算 Penalty Factor ---
    count_series = strat_df_internal[count_col]  # 从内部DF获取count列
    if count_series.empty or count_series.isnull().all():
         print(f"警告: '{count_col}' 列为空或全是 NaN。使用默认 penalty_factor 1.0。")
         auto_penalty_factor = 1.0
    else:
         median_count = count_series.median()
         if pd.isna(median_count) or median_count == 0:
             mean_count = count_series.mean()
             if pd.isna(mean_count) or mean_count == 0:
                 print(f"警告: '{count_col}' 的中位数和均值都为 0 或 NaN。Penalty factor 可能无效。使用 1.0。")
                 median_count = 1.0
             else:
                 median_count = mean_count
         auto_penalty_factor = abs(median_count * penalty_scaler)
         print(f"自动计算 Penalty Factor 基准 (count 中位数/均值): {median_count:.2f}")
         print(f"使用的 Penalty Factor (基准 * scaler): {auto_penalty_factor:.2f}")

    # --- 构建相关性查找字典 ---
    print("正在构建相关性查找字典...")
    corr_dict = {}
    correlation_value_col = 'Correlation'
    row1_col = 'Row1'
    row2_col = 'Row2'
    # 使用处理过的字符串ID构建字典
    for row in corr_df.itertuples(index=False):
        s1_str = getattr(row, row1_col)  # 已经是字符串且已去空格
        s2_str = getattr(row, row2_col)
        corr = getattr(row, correlation_value_col)
        if use_absolute_correlation:
            corr = abs(corr)
        key = tuple(sorted((s1_str, s2_str)))
        corr_dict[key] = corr
    print("相关性查找字典构建完成。")
    print(f"字典大小 (corr_dict): {len(corr_dict)}")  # 打印大小以供检查

    def get_correlation(s1: str, s2: str, lookup_dict: dict) -> float:
        """辅助函数：从字典中查找相关性 (输入为字符串ID)"""
        if s1 == s2:
            return 1.0
        key = tuple(sorted((s1, s2)))
        value = lookup_dict.get(key, 0.0)  # 缺失相关性默认为0
        return value

    # 获取所有有效策略的字符串ID (来自内部DF的索引)
    all_strategies_str = set(strat_df_internal.index)
    if not all_strategies_str:
         print("策略DataFrame内部处理后为空，无法选择。")
         return original_strat_df.iloc[0:0], corr_df.iloc[0:0]

    strat_df_internal[count_col] = pd.to_numeric(strat_df_internal[count_col], errors='coerce')
    strat_df_internal.dropna(subset=[count_col], inplace=True)
    all_strategies_str = set(strat_df_internal.index)  # 更新有效策略集合

    if not all_strategies_str:
         print("在处理 count 列后，没有有效的策略，无法选择。")
         return original_strat_df.iloc[0:0], corr_df.iloc[0:0]

    sorted_strategies_str = strat_df_internal.sort_values(count_col, ascending=False).index.tolist()

    if not sorted_strategies_str:
         print("排序后无有效策略，无法选择。")
         return original_strat_df.iloc[0:0], corr_df.iloc[0:0]

    # --- 2. 贪婪选择 ---
    selected_strategies_str = []  # 存储选中的策略的字符串ID
    candidate_pool_str = set(sorted_strategies_str)

    print(f"开始贪婪选择，目标数量 k={k}")

    # 选择第一个策略 (字符串ID)
    first_strategy_str = sorted_strategies_str[0]
    selected_strategies_str.append(first_strategy_str)
    candidate_pool_str.remove(first_strategy_str)

    # 设置相关性阈值：如果候选策略与任一已选策略的相关性超过该阈值，则不被考虑。
    correlation_threshold = 30

    while len(selected_strategies_str) < k and candidate_pool_str:
        best_candidate_str = None
        best_score = -np.inf

        # 遍历所有候选策略 (字符串ID)
        for candidate_str in candidate_pool_str:
            # 计算候选策略与已选策略之间的最大相关性
            max_corr_with_selected = 0.0
            if selected_strategies_str:
                current_max_corr = 0.0
                for selected_strat_str in selected_strategies_str:
                    corr = get_correlation(candidate_str, selected_strat_str, corr_dict)
                    current_max_corr = max(current_max_corr, corr)
                max_corr_with_selected = current_max_corr

            # 如果候选策略与任一已选策略的相关性超过阈值，则跳过该候选策略
            if max_corr_with_selected > correlation_threshold:
                continue

            candidate_count = strat_df_internal.loc[candidate_str, count_col]

            # 计算得分：在 count 的基础上扣除相关性惩罚
            score = candidate_count - auto_penalty_factor * max_corr_with_selected

            # 更新最佳候选
            if score > best_score:
                best_score = score
                best_candidate_str = candidate_str

        if best_candidate_str is None:
            print(f"  在第 {len(selected_strategies_str) + 1} 步无法找到合适的候选策略（满足相关性阈值要求），停止选择。")
            break

        # 添加最佳候选者 (字符串ID)
        selected_strategies_str.append(best_candidate_str)
        candidate_pool_str.remove(best_candidate_str)
        candidate_count = strat_df_internal.loc[best_candidate_str, count_col]
        original_id = strat_df_internal.loc[best_candidate_str, strategy_id_col]  # 获取原始ID用于打印
        # 计算并打印相关信息
        final_max_corr = 0.0
        if len(selected_strategies_str) > 1:
            current_max_corr = 0.0
            for s_str in selected_strategies_str[:-1]:
                corr = get_correlation(best_candidate_str, s_str, corr_dict)
                current_max_corr = max(current_max_corr, corr)
            final_max_corr = current_max_corr

    # --- 3. 从原始 DataFrame 中提取选定的策略 ---
    print(f"选择完成，共选出 {len(selected_strategies_str)} 个策略。")

    # 根据处理后的字符串ID筛选原始策略
    selected_mask = original_strat_df[strategy_id_col].astype(str).str.strip().isin(selected_strategies_str)
    selected_strategies_df_unordered = original_strat_df[selected_mask].copy()

    # 保证输出的顺序与选择顺序一致
    if not selected_strategies_df_unordered.empty and selected_strategies_str:
        id_map = selected_strategies_df_unordered.set_index(selected_strategies_df_unordered[strategy_id_col].astype(str).str.strip())
        selected_strategies_df = id_map.loc[selected_strategies_str].copy()
        selected_strategies_df.reset_index(drop=True, inplace=True)
    else:
        selected_strategies_df = selected_strategies_df_unordered

    selected_strategies_set_str = set(selected_strategies_str)

    corr_filter_mask = corr_df[row1_col].isin(selected_strategies_set_str) & \
                       corr_df[row2_col].isin(selected_strategies_set_str)
    selected_correlation_df = corr_df[corr_filter_mask].copy()

    # 尝试将相关性 DataFrame 中的 Row1/Row2 恢复为原始类型
    original_id_dtype = original_strat_df[strategy_id_col].dtype
    if not pd.api.types.is_string_dtype(original_id_dtype):
        id_str_to_original_map = {}
        for _idx, row in original_strat_df.drop_duplicates(subset=[strategy_id_col], keep='first').iterrows():
            id_str = str(row[strategy_id_col]).strip()
            id_orig = row[strategy_id_col]
            id_str_to_original_map[id_str] = id_orig

        try:
            selected_correlation_df[row1_col] = selected_correlation_df[row1_col].map(id_str_to_original_map)
            selected_correlation_df[row2_col] = selected_correlation_df[row2_col].map(id_str_to_original_map)
            selected_correlation_df.dropna(subset=[row1_col, row2_col], inplace=True)
            selected_correlation_df[row1_col] = selected_correlation_df[row1_col].astype(original_id_dtype)
            selected_correlation_df[row2_col] = selected_correlation_df[row2_col].astype(original_id_dtype)
            print(f"相关性DataFrame中的ID已尝试恢复为原始类型: {original_id_dtype}")
        except Exception as e:
            print(f"警告：尝试将相关性DF中的ID转回原始类型时出错: {e}。将返回字符串形式的ID。")

    # --- 5. 返回结果 ---
    return selected_strategies_df, selected_correlation_df

def compute_signal(df, col_name):
    """
    根据历史行情数据(df)和指定信号名称(col_name)，生成交易信号和对应目标价格。

    说明：
      - 信号的目标价格不再使用 clip() 调整，
        而是在判断目标价格是否落在当前bar的 low 和 high 区间内，
        若目标价格超出区间，则认为信号无效（不产生信号）。
      - 当前支持的信号类型包括：
          - abs: 绝对百分比突破信号
              示例："abs_20_2_long" (20周期内最低价向上2%多头突破)
          - relate: 相对区间百分比突破信号
              示例："relate_20_50_short" (20周期区间顶部向下50%空头突破)
          - donchian: 唐奇安通道突破信号（实时价格触发优化）
              示例："donchian_20_long" (20周期最高价向上突破多头信号)
          - boll: 布林带信号
              示例："boll_20_2_long" 或 "boll_20_2_short"
          - macross: MACROSS 信号 (双均线交叉信号)
              示例："macross_10_20_long"
          - rsi: RSI 超买超卖反转信号
              示例："rsi_14_70_30_long"
          - macd: MACD交叉信号
              示例："macd_12_26_9_long"
          - cci: 商品通道指数超买超卖反转信号
              示例："cci_20_short"
              （若传入参数不足，则采用默认常数0.015）
          - atr: ATR波动率突破信号
              示例："atr_14_long"

    参数:
      df: pandas.DataFrame，必须包含以下列：
          "close": 收盘价
          "high": 最高价
          "low": 最低价
      col_name: 信号名称，格式如 "signalType_param1_param2_..._direction"

    返回:
      tuple:
        - signal_series: pandas.Series(bool)，当满足信号条件时为 True，否则为 False。
        - trade_price_series: pandas.Series(float)，信号触发时建议的目标交易价格；
          若目标价格超出当前bar的 low 和 high，则不产生信号。
    """

    parts = col_name.split('_')
    signal_type = parts[0]
    direction = parts[-1]

    if signal_type == 'abs':
        period = int(parts[1])
        abs_value = float(parts[2]) / 100
        if direction == "long":
            min_low_series = df['low'].shift(1).rolling(period).min()
            target_price = (min_low_series * (1 + abs_value)).round(4)
            signal_series = df['high'] > target_price
        else:
            max_high_series = df['high'].shift(1).rolling(period).max()
            target_price = (max_high_series * (1 - abs_value)).round(4)
            signal_series = df['low'] < target_price

        # 检查目标价格是否落在当前bar的low与high之间
        valid_price = (target_price >= df['low']) & (target_price <= df['high'])
        signal_series = signal_series & valid_price
        trade_price_series = target_price  # 直接使用计算的目标价格

        # 可选调试记录
        df['target_price'] = target_price
        df['signal_series'] = signal_series
        df['trade_price_series'] = trade_price_series
        return signal_series, trade_price_series

    elif signal_type == 'relate':
        period = int(parts[1])
        percent = float(parts[2]) / 100
        min_low_series = df['low'].shift(1).rolling(period).min()
        max_high_series = df['high'].shift(1).rolling(period).max()
        if direction == "long":
            target_price = (min_low_series + percent * (max_high_series - min_low_series)).round(4)
            signal_series = df['high'] > target_price
        else:
            target_price = (max_high_series - percent * (max_high_series - min_low_series)).round(4)
            signal_series = df['low'] < target_price

        valid_price = (target_price >= df['low']) & (target_price <= df['high'])
        signal_series = signal_series & valid_price
        trade_price_series = target_price
        return signal_series, trade_price_series

    elif signal_type == 'donchian':
        period = int(parts[1])
        if direction == "long":
            highest_high = df['high'].shift(1).rolling(period).max()
            signal_series = df['high'] > highest_high
            target_price = highest_high
        else:
            lowest_low = df['low'].shift(1).rolling(period).min()
            signal_series = df['low'] < lowest_low
            target_price = lowest_low

        valid_price = (target_price >= df['low']) & (target_price <= df['high'])
        signal_series = signal_series & valid_price
        trade_price_series = target_price.round(4)
        return signal_series, trade_price_series

    elif signal_type == 'boll':
        period = int(parts[1])
        std_multiplier = float(parts[2])
        ma = df['close'].rolling(window=period, min_periods=period).mean()
        std_dev = df['close'].rolling(window=period, min_periods=period).std()
        upper_band = (ma + std_multiplier * std_dev).round(4)
        lower_band = (ma - std_multiplier * std_dev).round(4)
        if direction == "long":
            signal_series = (df['close'].shift(1) < lower_band.shift(1)) & (df['close'] >= lower_band)
        else:  # short
            signal_series = (df['close'].shift(1) > upper_band.shift(1)) & (df['close'] <= upper_band)
        # 此处直接返回收盘价作为交易价格
        return signal_series, df["close"]

    elif signal_type == 'macross':
        fast_period = int(parts[1])
        slow_period = int(parts[2])
        fast_ma = df["close"].rolling(window=fast_period, min_periods=fast_period).mean().round(4)
        slow_ma = df["close"].rolling(window=slow_period, min_periods=slow_period).mean().round(4)
        if direction == "long":
            signal_series = (fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma >= slow_ma)
        else:
            signal_series = (fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma <= slow_ma)
        trade_price = df["close"]
        return signal_series, trade_price

    elif signal_type == 'rsi':
        period = int(parts[1])
        overbought = int(parts[2])
        oversold = int(parts[3])
        delta = df['close'].diff(1).astype(np.float32)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        # 防止除0错误
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        if direction == "long":
            signal_series = (rsi.shift(1) < oversold) & (rsi >= oversold)
        else:
            signal_series = (rsi.shift(1) > overbought) & (rsi <= overbought)
        return signal_series, df['close']

    elif signal_type == 'macd':
        fast_period, slow_period, signal_period = map(int, parts[1:4])
        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        if direction == "long":
            signal_series = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line >= signal_line)
        else:
            signal_series = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line <= signal_line)
        return signal_series, df["close"]

    elif signal_type == 'cci':
        period = int(parts[1])
        # 若参数不足，采用默认常数0.015
        if len(parts) == 3:
            constant = 0.015
        else:
            constant = float(parts[2]) / 100
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        cci = (tp - ma) / (constant * md)
        if direction == "long":
            signal_series = (cci.shift(1) < -100) & (cci >= -100)
        else:
            signal_series = (cci.shift(1) > 100) & (cci <= 100)
        return signal_series, df['close']

    elif signal_type == 'atr':
        period = int(parts[1])
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        atr_ma = atr.rolling(period).mean()
        if direction == "long":
            signal_series = (atr.shift(1) < atr_ma.shift(1)) & (atr >= atr_ma)
        else:
            signal_series = (atr.shift(1) > atr_ma.shift(1)) & (atr <= atr_ma)
        return signal_series, df['close']

    else:
        raise ValueError(f"未知信号类型: {signal_type}")

def get_config(key):
    """
    从 config.json 文件中获取指定字段的值
    :param key: 配置字段名
    :return: 配置字段值
    """
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接 config.json 文件的绝对路径
    config_file = os.path.join(base_dir, 'config.json')

    # 检查 config.json 文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件 '{config_file}' 不存在，请检查文件路径。")

    # 读取配置文件
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件 '{config_file}' 格式错误: {e}")

    # 获取指定字段的值
    if key not in config_data:
        raise KeyError(f"配置文件中缺少字段: {key}")

    return config_data[key]


import copy
import re


# --- 辅助函数 (time_to_ms 保持不变，ms_to_time 修改) ---

def time_to_ms(time_input: str | float | int) -> int:
    """
    将多种时间格式统一转换为毫秒。
    该函数非常稳健，可以处理以下格式：
    - 数字 (int/float): 12.345 (代表秒)
    - 纯秒数字符串: "12.345"
    - 标准SRT时间码: "00:01:02,345" 或 "00:01:02.345"
    - 省略小时的时间码: "01:02.345"
    - 只有分秒的时间码: "03.482"

    Args:
        time_input: 多种格式的时间输入。

    Returns:
        总毫秒数 (int)。
    """
    if isinstance(time_input, (int, float)):
        return int(time_input * 1000)

    time_str = str(time_input).strip()

    try:
        return int(float(time_str.replace(',', '.')) * 1000)
    except ValueError:
        pass

    time_str = time_str.replace(',', '.')

    parts = time_str.split(':')
    h, m, s = 0, 0, 0.0

    try:
        if len(parts) == 3:  # HH:MM:SS.ms
            h = int(parts[0])
            m = int(parts[1])
            s = float(parts[2])
        elif len(parts) == 2:  # MM:SS.ms
            m = int(parts[0])
            s = float(parts[1])
        elif len(parts) == 1:  # SS.ms
            s = float(parts[0])
        else:
            raise ValueError("时间码中的冒号过多")

        return int((h * 3600 + m * 60 + s) * 1000)

    except (ValueError, IndexError):
        raise ValueError(f"无法解析的时间格式: '{time_input}'")


def ms_to_time(ms: int) -> str:
    """将毫秒转换为'HH:MM:SS.ms'格式的时间字符串。"""
    if ms < 0: ms = 0
    s, ms_rem = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    # --- 修改点：将逗号改为点 ---
    return f"{h:02d}:{m:02d}:{s:02d}.{ms_rem:03d}"


# --- 主函数 (无需修改) ---

def optimize_subtitle_timing(subtitle_list: list) -> list:
    """
    优化字幕列表的时间占用，调整间隔，记录移动时间，并计算持续时长。

    Args:
        subtitle_list: 存放字幕信息的字典列表。

    Returns:
        处理后的新字幕列表，增加了 duration 字段，并且所有时间戳统一为 'HH:MM:SS.ms' 格式。
    """
    if not subtitle_list:
        return []

    processed_list = copy.deepcopy(subtitle_list)

    for segment in processed_list:
        segment['old_startTime'] = segment['startTime']
        segment['old_endTime'] = segment['endTime']
        segment['forward_shift_ms'] = 0
        segment['backward_shift_ms'] = 0

    first_segment = processed_list[0]
    start_ms = time_to_ms(first_segment['startTime'])
    if start_ms > 0:
        new_start_ms = start_ms // 2
        shift = start_ms - new_start_ms
        first_segment['startTime'] = ms_to_time(new_start_ms)
        first_segment['forward_shift_ms'] = shift

    for i in range(len(processed_list) - 1):
        current_segment = processed_list[i]
        next_segment = processed_list[i + 1]

        current_end_ms = time_to_ms(current_segment['endTime'])
        next_start_ms = time_to_ms(next_segment['startTime'])

        if next_start_ms < current_end_ms:
            next_start_ms = current_end_ms

        gap_ms = next_start_ms - current_end_ms

        if gap_ms > 0:
            adjustment_ms = gap_ms / 2
            max_adjustment_ms = 500
            actual_adjustment_ms = min(adjustment_ms, max_adjustment_ms)

            new_current_end_ms = current_end_ms + actual_adjustment_ms
            current_segment['endTime'] = ms_to_time(int(new_current_end_ms))

            new_next_start_ms = next_start_ms - actual_adjustment_ms
            next_segment['startTime'] = ms_to_time(int(new_next_start_ms))

            current_segment['backward_shift_ms'] += actual_adjustment_ms
            next_segment['forward_shift_ms'] += actual_adjustment_ms

    for segment in processed_list:
        segment['forward_shift_ms'] = int(segment['forward_shift_ms'])
        segment['backward_shift_ms'] = int(segment['backward_shift_ms'])

    for segment in processed_list:
        start_ms = time_to_ms(segment['startTime'])
        end_ms = time_to_ms(segment['endTime'])

        segment['startTime'] = ms_to_time(start_ms)
        segment['endTime'] = ms_to_time(end_ms)

        duration_seconds = (end_ms - start_ms) / 1000.0
        segment['duration'] = round(duration_seconds, 3)

    return processed_list

def merge_time_segments(segments: list) -> list:
    """
    合并列表中相邻且连续的时间段。

    Args:
        segments (list): 一个包含时间段字典的列表。
                         每个字典需包含 'original_start_time' 和 'original_end_time'。
                         列表应预先按开始时间排序。

    Returns:
        list: 一个新的列表，其中包含了合并后的时间段。
    """
    # 如果列表为空或只有一个元素，无需合并，直接返回副本
    if not segments or len(segments) < 2:
        return segments[:]

    # 用于存放最终结果的列表
    merged_list = []

    # 将第一个时间段作为起始合并段，注意使用 .copy() 以免修改原始输入
    current_merged_segment = segments[0].copy()

    # 从第二个元素开始遍历
    for i in range(1, len(segments)):
        next_segment = segments[i]

        # 检查当前合并段的结束时间是否与下一个段的开始时间完全相同
        if current_merged_segment['original_end_time'] == next_segment['original_start_time']:
            # --- 条件满足，进行合并 ---

            # 1. 更新结束时间
            current_merged_segment['original_end_time'] = next_segment['original_end_time']

            # 2. 合并其他描述性信息（这里我们用 '&' 连接场景ID，用换行和分隔符连接理由）
            current_merged_segment['scene_id'] += f" & {next_segment['scene_id']}"
            current_merged_segment[
                'reasoning'] += f"\n\n--- [合并自 {next_segment['scene_id']}] ---\n{next_segment['reasoning']}"

            # new_sequence_index 保持为合并段的第一个索引，所以不做任何操作
        else:
            # --- 条件不满足，无法合并 ---

            # 1. 将已经完成的合并段添加到结果列表
            merged_list.append(current_merged_segment)

            # 2. 将当前遍历到的这个时间段作为新的起始合并段
            current_merged_segment = next_segment.copy()

    # 循环结束后，不要忘记将最后一个正在处理的 current_merged_segment 添加到结果中
    merged_list.append(current_merged_segment)

    return merged_list


def read_json(json_path):
    """
    读取 JSON 文件并返回内容。

    Args:
        json_path (str): JSON 文件的路径。

    Returns:
        dict: 解析后的 JSON 内容。
    """
    if not os.path.exists(json_path):
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析 JSON 文件 '{json_path}': {e}")

def save_json(json_path, data):
    """
    将数据保存为 JSON 文件。

    Args:
        json_path (str): 要保存的 JSON 文件路径。
        data (dict): 要保存的数据。
    """
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



def fill_time_gaps(segments: list) -> list:
    """
    填充列表中时间轴的断层。

    Args:
        segments (list): 包含时间段字典的原始列表。

    Returns:
        list: 填充了时间间隙并重新排序ID后的新列表。
    """
    if not segments or len(segments) < 2:
        return segments

    # 为防止原始列表顺序不正确，先根据 startTime 排序
    # (虽然你的例子是按顺序的，但这是一个好习惯)
    segments.sort(key=lambda x: time_to_ms(x['startTime']))

    new_segments = []
    # 遍历列表，检查相邻两个元素之间是否存在间隙
    for i in range(len(segments) - 1):
        current_segment = segments[i]
        next_segment = segments[i+1]

        # 1. 首先将当前元素添加到新列表中
        new_segments.append(current_segment)

        # 2. 比较当前元素的 endTime 和下一个元素的 startTime
        end_time_current = current_segment['endTime']
        start_time_next = next_segment['startTime']

        if end_time_current != start_time_next:
            # 发现时间断层，创建一个新的字典来填充

            gap_start_ms = time_to_ms(end_time_current)
            gap_end_ms = time_to_ms(start_time_next)
            gap_duration_ms = gap_end_ms - gap_start_ms

            print(f"发现时间断层: 从 {end_time_current} 到 {start_time_next} {gap_duration_ms}")


            # 创建一个表示无声间隙的新字典
            gap_segment = {
                "id": 0,  # ID 稍后会统一重新编号
                "startTime": end_time_current,
                "endTime": start_time_next,
                "text": "[无声]",  # 使用特殊文本标记这是填充的间隙
                "optimizedText": "[无声]",
                "old_startTime": None,
                "old_endTime": None,
                "forward_shift_ms": 0,
                "backward_shift_ms": 0,
                "duration": gap_duration_ms / 1000.0,
                "outputPath": None,
                "trimmedDuration": gap_duration_ms / 1000.0
            }
            new_segments.append(gap_segment)

    # 3. 添加原始列表中的最后一个元素
    new_segments.append(segments[-1])

    # 4. 重新为所有元素编号，确保 ID 是连续的
    for index, segment in enumerate(new_segments):
        segment['id'] = index + 1

    return new_segments

def find_file_by_name(root_dir: str, target_filename: str) -> str | None:
    """
    在指定目录（包括子目录）中查找指定文件名的文件。

    参数：
    - root_dir: 要搜索的起始目录
    - target_filename: 要查找的文件名（完全匹配）

    返回：
    - 找到时返回完整文件路径；找不到时返回 None
    """
    for dirpath, _, filenames in os.walk(root_dir):
        if target_filename in filenames:
            return os.path.join(dirpath, target_filename)
    return None

def format_seconds_to_mmss(seconds: float) -> str:
    """将总秒数格式化为 'MM:SS' 字符串。"""
    if seconds is None or seconds < 0:
        return "00:00"
    total_seconds = int(round(seconds))  # 四舍五入到最近的整数秒
    minutes = total_seconds // 60
    seconds_part = total_seconds % 60
    return f"{minutes:02d}:{seconds_part:02d}"