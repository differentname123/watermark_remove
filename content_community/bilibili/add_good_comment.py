import datetime
import multiprocessing
import os
import random
import re
import time
import traceback
from typing import Any, Dict, List, Optional

from LLM.gemini import get_llm_content
from common_utils.common_utils import read_json, get_config, save_json_safe, init_config, process_product_title
from content_community.bilibili.add_good_comment_kouling import search_goods
from content_community.bilibili.bili_utils import fetch_goods, update_short_url, list_selection_car_items
from content_community.bilibili.comment import BilibiliCommenter
from content_community.bilibili.get_comment import get_bilibili_comments
from content_community.bilibili.get_danmu import string_to_list
from content_community.bilibili.high_quality_hudong import find_video_by_bvid
from common_utils.common_utils import string_to_object, save_json_safe, read_json
from content_community.taobao.taobao_utils import add_to_favorites, creat_and_favorite

BASE_DIR = 'goods_info'


# success_bvids_file = f"{BASE_DIR}/all_goods_bvid.json"

base_prompt = """

## 角色 (Role)

你是一位顶级社交媒体电商运营专家及数据分析师。你的核心能力是在深刻洞察内容与观众情感的基础上，**平衡“内容原生性”与“市场转化效率”**，以推荐最具商业价值的商品。

## 核心任务 (Task)

根据我提供的、关于一个短视频的完整JSON数据，为其推荐多种最适合推广的商品。你的分析必须逻辑严谨、数据驱动，并以结构清晰、信息丰富的JSON格式输出最终结果。

## 输入数据说明 (Input Data)

我将提供一个JSON对象，其中包含三大关键信息模块：

1.  `titles`: 视频的多角度营销包装方案。
2.  `video_anlyse`: 对视频内容的客观结构化分析。
3.  `comments`: 真实观众的热门评论，**这是判断用户真实需求和情绪风向的核心依据**。

## 处理指令 (Processing Instructions)

你的分析过程必须严格遵循以下**四大核心原则**：

**1. 转化优先过滤器 (Conversion-First Filter)**

  * **A. 默认规则 (Default Rule):** **将95%的注意力集中在**高频次、低决策成本、适合冲动消费的**快消品和大众化商品**上。这是确保高转化率的基本盘。
      * **正面清单 (Priority List):** 食品饮料、美妆护肤、创意日用、趣味玩具、应季爆款小商品、数码配件、宠物用品等。
  * **B. 例外条款 (Exception Clause):** 在极少数情况下，你可以推荐“默认规则”之外的商品（如书籍、课程、小众商品等）。**但必须满足以下所有严苛条件：**
      * **① 证据确凿:** `comments` 或 `video_anlyse` 中必须存在**大量、明确、直接**指向该特定商品的讨论或需求。例如，评论区大量用户在问“视频里这本书叫什么名字？”或“这个软件哪里下载？”。
      * **② 不可替代:** 该商品与视频内容的绑定是独一无二的，无法用一个更大众化的商品轻易替代。
      * **③ 理由阐述:** 若使用此条款，必须在`reason`字段中明确指出**“基于例外条款推荐，证据源于评论区高频询问”**，以供人工审核。

**2. 强关联性原则 (Strong Relevance)**

  * 商品必须与视频的核心**场景**（如办公室、饭局）、**主题**（如情侣、职场）、**情绪**（如尴尬、治愈、搞笑）或**高光梗**（如“原文是条狗”）有直接、巧妙的联系。

**3. 解决方案式匹配 (Solution-Oriented Matching)**

  * 站在观众视角思考：这个商品是否为我在视频中感受到的某种情绪或遇到的场景，提供了一个**“解决方案”**？
      * **情绪方案:** 通过美食获得快乐、用解压玩具缓解焦虑。
      * **场景方案:** 用破冰游戏活跃聚会气氛、用便携漱口水化解饭后尴尬。

**4. 精准关键词策略 (Precision Keyword Strategy)**

  * 为每个商品生成高商业价值的搜索关键词。关键词的设计应遵循**“组合公式”**，以提升搜索精准度和转化率，关键词不能够出现空格或者'/'等其它符号：
      * **公式A (品类+特性):** 例如 `“漱口水便携”`、`“速溶咖啡提神”`
      * **公式B (场景/人群+品类):** 例如 `“办公室零食”`、`“情侣礼物”`
      * **公式C (视频热梗/情绪+品类):** 例如 `“社死神器”`、`“解压玩具”`
      * **一个商品的关键词组合应尽可能覆盖不同公式**，形成搜索矩阵。

## 输出格式要求 (Output Format Requirement)

你的回答**必须是唯一且纯粹的JSON对象**。禁止在JSON代码块前后添加任何Markdown标记、介绍、总结或任何形式的解释性文字。输出的根节点必须是一个名为 `product_recommendations` 的数组，数组中的每个对象都必须严格遵循以下**四个字段**的规范：

```json
{
  "product_recommendations": [
    {
      "product_name": "string",
      "reason": "string",
      "score": "integer",
      "keywords": ["string"]
    }
  ]
}
```

### **JSON 字段详细解释:**

  * `product_name`: (string) **具体的商品名称或品类。** 例如: `"三只松鼠每日坚果"`, `"usmile便携漱口水"`, `"桌面小型加湿器"`。**名称应简短且具有代表性。**
  * `reason`: (string) **简明扼要的核心推荐理由。** 必须直接关联视频内容、高光梗或观众情绪。**如果应用了【例外条款】，必须在此处注明。**
  * `score`: (integer) **一个 1-10 的综合推荐指数。** 该分数是基于**转化潜力（权重70%）**和**内容相关度（权重30%）**的加权评估。应用【例外条款】的商品，其转化潜力需经过更审慎的评估。
  * `keywords`: (array of strings) **一个包含核心搜索关键词的字符串数组。** 必须遵循**【精准关键词策略】**生成，兼具概括性和搜索热度。

"""


final_prompt = """
你是一个以“最大化下单转化率”为北极星的**转化导向电商策略师 + 病毒式评论操盘手**。收到一个短视频 JSON（包含 fields: titles, video_anlyse, comments, goods）后，按下述流程执行并**只返回符合输出 Schema 的纯 JSON**（不要额外文字）。

【总体目标】

  - 在不损害佣金潜力的前提下，最大化下单转化率，用最少的推荐位、最高的命中率，把商品筛成“必带上车”的推荐，并为每件入选商品生成高转化的置顶神评与 10–15 条链式助推评论，形成买家信任链与购买冲动。

【输入说明】

  - `titles`: 数组或字符串（多角度标题）
  - `video_anlyse`: 结构化分析（场景、矛盾、台词、节奏、高光）
  - `comments`: 真实观众评论数组（用于情绪与需求挖掘）
  - `goods`: 候选商品数组。字段可能不统一，但**每项至少包含 `outerId` 和 `goodsName`**。常见字段有 `brand`, `shopName`, `promo_price`, `leaf_category`, `coupon_value` 等。

-----

### **【执行步骤（必须严格执行）】**

**1. 输入预处理与标准化（首要步骤）**
在进行洞察分析前，必须先对每个 `goods` 对象执行以下标准化流程，以兼容不规范的数据输入。所有依据本规则生成或估算的信息，必须在最终输出的 `reason` 字段中透明标注来源。

  - **`description` (描述) 生成**: 若原始数据中不存在 `description` 字段，则通过拼接现有字段合成一个：`"{goodsName}；类目：{leaf_category}；品牌：{brand}"`。
  - **`shop_official_flag` (官方店标志) 判断**: 若 `shopName` 包含“官方/旗舰/直营/官方旗舰”，则在内存中创建一个临时标志 `shop_official_flag=true`。
  - **`has_coupon` (优惠券标志) 判断**: 若 `coupon_value` 存在且不为空/0，则在内存中创建一个临时标志 `has_coupon=true`。
  - **`is_mentioned_in_comments` (评论区提及标志) 判断**: 扫描 `comments`，若 `goodsName` 或其核心词被提及 ≥1 次，则创建一个临时标志 `is_mentioned_in_comments=true`。

**2. 深度洞察（必做）**

  - 从 `video_anlyse` + `comments` 提取：核心冲突/欲望（1句），高光梗/金句（1-2项），主流情绪（正/负/中 比例或定性）。
  - 标注可能的购买扳机（如省时、省钱、面子、好玩、社交货币、治愈等）。

**3. 构建转化假设赛道（必做）**

  - 基于洞察，构思 2–4 个“转化假设”（每个为一句话，例如“痛点解决”、“身份认同”、“场景复现”、“梗参与”）。
  - 每个赛道目标明确、可衡量（应能用商品属性直接验证）。

**4. 数据驱动筛选（必做 — 采用主备逻辑）**
对每个 `goods`（经过标准化处理后）计算四项子分（0–10）：

  * **Relevance（相关度，权重 40%）**: 基于 `goodsName`, `leaf_category`, 以及合成的 `description` 与转化赛道匹配程度进行量化（1-10）。

  * **Commercial（商业潜力，权重 30%）**: **采用主备用方案**进行评分，确保结果的稳定与准确。

      * **主方案 (基于价格模型)**: 当 `promo_price` 存在且有效时启用。
          * **附加分 (在基础分上累加)**:
              * **品牌背书 (+1分)**: `shop_official_flag` 为 true。
              * **促销信号 (+1分)**: `has_coupon` 为 true。
      * **备用方案 (基于启发式估算)**: 当 `promo_price` 缺失时启用。
          * **消费需求层级 (+5分)**: `leaf_category` 或 `goodsName` 属于高频消费品（零食、日用等）。
          * **热度信号 (+3分)**: `goodsName` 包含“热卖/爆款/推荐/畅销”。
          * **品牌背书 (+1分)**: `shop_official_flag` 为 true。
          * **社交流量信号 (+2分)**: `is_mentioned_in_comments` 为 true。
          * **基础分**: 若无任何上述信号，则**基础分为 4 分**。
      * **计算方式**: 采用所选方案，将各项得分相加，并将**最终结果裁剪到 0–10 的区间内**。

  * **SocialProof（社证明，权重 20%）**: 优先依据 `is_mentioned_in_comments` 标志和评论内容。

      * `comments` 中 ≥2 条正面提及 → 7–9分
      * 1 条正面提及 (`is_mentioned_in_comments`=true) → 5–6分
      * 无提及，但 `goodsName` 含“推荐/热卖” → 5分

  * **Diversity（差异化，权重 10%）**: 基于 `leaf_category` 判断。首个出现的品类得10分，后续重复的品类逐步递减。

**合成公式（不变）**: `raw = 0.4*Relevance + 0.3*Commercial + 0.2*SocialProof + 0.1*Diversity`；`score = round(raw)`（取 1–10）。
**透明审计**: 若评分中使用了任何标准化的估算数据，必须在输出的 `reason` 中追加来源，例如“**（估算来源：goodsName/shopName）**”。

**5. 回退与风控（必做）**

  - 若所有商品 `score` ≤ 5，则仍输出综合最高项，但在 `reason` 中写明“放宽匹配标准”。
  - 禁止推荐处方药；涉及保健/药品必须标注“非处方/保健，建议咨询专业人士”，并避免疗效断言。
  - 避免明显与视频人设/评论氛围相冲突的推荐（若冲突，降低 `score` 并在 `reason` 说明）。

**6. 创作文案（必须为每件入选商品生成）**
    - **pinned_comment（置顶神评）**
      * 目标：创作一条本身就极具“点赞、转发、回复”潜力的神评，优先制造情绪共鸣、好笑/好奇或强烈认同感，不要直接以带货为主。带货意图应隐晦或完全不显现，留给下方的 shill_comments 逐步接力。
      * 字数与风格：严格 ≤60 个中文字符（建议 40–55 字以提升易读与传播性）；第一人称或矿工式观察句；口语化、节奏感强；可使用 0–1 个 emoji，但避免多重广告语。
      * 结构建议（非机械公式，给创作灵感）：
        1. **钩子句（1-2 短句）**：一句让人停下来的观察或反转（惊讶/怀疑/自嘲/共鸣）。
        2. **个性化句（1 短句）**：用“我”或“我们”立场加强代入感（可以是夸张、凡尔赛或悔改式）。
        3. **留白句（0–1 短句）**：制造悬念或抛出开放式问题，诱导回复和转发。
      * 禁忌与底线：不得出现显性促销用语（“买/下单/点链接/秒杀”），不得虚假夸大或违规内容。
      * 建议：为A/B测试生成 2 个风格变体（例如“幽默型”与“共情型”），最终输出你觉得好的那个评论。
      * **【输出要求】**：**严格遵守！** 风格变体的说明仅用于指导创作，**最终输出的 `pinned_comment` 字符串中，绝对不能包含如“[幽默型]”、“[共情型]”等任何形式的风格标签**。它必须是一个纯粹的评论文本。


    - shill_comments（低介入自然讨论策略）
      * 目的：在神评引发互动后，以克制、非引导式的信息补充维持真实讨论氛围。不得出现购买导向或 @。
      * 语气与限制：
        - 禁止：@、链接/二维码、价格/折扣/券、直指购买渠道（店铺名、跳转指引）、以及“买/下单/冲/安排/点链接/领券/必入”等显性引导词。
        - 禁止虚构购买/使用经历；如为合作或样品体验，需在文案中显式标注（例如：含合作/样品体验）。
        - 建议每条 ≤40 字；以感受/观察/提问为主，避免命令句、口播式话术。
      * 角色与脚本（最好覆盖全部6个角色）：
        1. **体验派（克制认可）**：第一人称轻描淡写的使用感，不做效果承诺，不提供购买线索。
        2. **氛围烘托型（制造热度与从众心理）**：核心任务是表达强烈的拥有欲或暗示已经采取了购买相关的行动，但必须避免直接说“我买了”或“已下单”等直白字眼，旨在创造一种“很多人都想要”的群体情绪。
        3. **好奇提问型（引出产品细节与证据）**：扮演一个感兴趣但持有疑虑的潜在买家。评论应针对产品的某个具体方面（如效果、耐用性、性价比等）提出明确的问题。
        4. **体验分享型（以“过来人”身份建立信任）**：内容上要分享真实、具体的使用感受。为了最大化可信度，可以适度提及一些微不足道的小缺点或使用中的注意事项。
        5. **理性参考**：补充客观信息（材质/参数/适用场景），不出现价格与渠道信息。
        6. **场景代入**：描述更合适的使用情境或人群，避免诱导行动。
      * **【输出要求】**：**严格遵守！** 此处的角色与脚本仅用于指导你创作评论的【思路和角度】，**最终输出的每一条 `shill_comments` 字符串中，绝对不能包含如“[体验派]”、“[氛围烘托型]”等任何形式的分类标签或前缀**。输出内容必须是纯粹、自然的评论文本本身。


**7. 工程化输出（严格）**

  - 输出根节点：`product_recommendations`（数组，按 `score` 从高到低排序）。
  - 每个推荐对象字段如下（必须全部包含）：
      * `outerId` (string) — 原样返回
      * `goodsName` (string) — 原样返回
      * `reason` (string) — 一句话核心推荐理由，包含驱动评分的核心证据（如“9.9元低价”、“零食类目高频消费”）与所命中的“转化假设”，必要时附带估算来源。
      * `score` (integer) — 1–10
      * `keywords` (array[string]) — 3–5 个高意向搜索词（从 `goodsName`, `leaf_category`, `brand` 中提炼）。
      * `estimated_ctr` (float) — 预计点击转化率，**使用以下可复现公式计算**：
        ```
        base_ctr = 0.03
        score_factor = 0.07 * (score / 10)
        promo_factor = 0.02 if (promo_price is not None and promo_price <= 20) else 0
        estimated_ctr = min(base_ctr + score_factor + promo_factor, 0.5)
        ```
      * `pinned_comment` (string) — ≤ 60 中文字符
      * `shill_comments` (array[string]) — 10–15 条链式助推评论
  - **严格要求：输出只含此 JSON 对象，不可附加任何额外文本或解释。**
【最后的风控提醒（必须遵守）】
- 不得推广处方药、未成年人性化内容、违法或仇恨内容。
- 对保健与金融类商品避免绝对化承诺与疗效/收益保证。
"""

def search_local_goods_info(key_word, local_file_path, max_time_diff=60* 60 * 24 * 7):
    """
    本地查询商品信息,更新时间在 max_time_diff 秒内的商品信息。
    """
    result_goods = []
    current_time = time.time()
    expire_count = 0
    local_good_data = read_json(local_file_path)
    # 遍历local_data_list，查找goodsName包含key_word的商品信息
    for key, good in local_good_data.items():
        if key_word in good['goodsName']:
            # 检查更新时间是否在max_time_diff秒内
            if current_time - good.get('updateTime', 0) <= max_time_diff:
                result_goods.append(good)
            else:
                expire_count += 1
    # 打印详细的总体信息
    # print(f"通过本地文件 关键词 '{key_word}'结果：找到 {len(result_goods)} 条商品信息，过期商品数量：{expire_count} 总共 {len(local_good_data)} 条商品信息。 目标文件：{local_file_path}")
    return result_goods

def search_goods_info(key_word_list, user_name='ruru'):
    """
    根据关键词列表抓取商品信息。
    """
    print(f"\n\n正在抓取商品信息，关键词列表长度 {len(key_word_list)} 关键词列表：{key_word_list}")
    property_goods = []
    current_time = time.time()

    local_file_path = f"{BASE_DIR}/{user_name}_goods_info.json"
    for key_word in key_word_list:
        goods = search_local_goods_info(key_word, local_file_path)
        if len(goods) < 1:
            goods = fetch_goods(get_config(f'{user_name}_bilibili_total_cookie'), 50, key_word)
            # print(f"通过接口 关键词 '{key_word}' 抓取到 {len(goods)} 条商品信息。")
            # 更新local_file_path的数据
            if goods:
                local_good_data = read_json(local_file_path)
                for good in goods:
                    good['updateTime'] = current_time
                    outerId = good['outerId'].split('-')[-1]
                    # good['outerId'] = outerId
                    local_good_data[outerId] = good
                save_json_safe(local_file_path, local_good_data)

        property_goods.extend(search_local_goods_info(key_word, local_file_path))
    # 去重
    property_goods = {good['outerId']: good for good in property_goods}.values()

    # 打印详细的总体信息
    print(f"抓取到 {len(property_goods)} 条商品信息。key_word_list {len(key_word_list)} 关键词列表 {key_word_list} ")
    return {good['outerId'].split('-')[-1]: good for good in property_goods}


def format_title(raw_data: dict) -> dict:
    """
    处理原始视频信息，提取和组织用于商品推荐的关键字段。

    该函数会执行以下操作：
    1. 遍历输入的每个视频方案。
    2. 提取 '定位', '标题', '标签', '分区编号' 等核心字段。
    3. 将 '简介' 中的 '核心看点' 和 '价值承诺' 合并为 '摘要'。
    4. 删除如 '设计策略', '封面', '优势', '增长潜力' 等与商品推荐无关的元信息。
    5. 重命名 '分区编号' 为 '分区ID' 以提高可读性。
    6. 返回一个结构清晰、只包含推荐所需信息的新字典。

    :param raw_data: 包含多个视频方案的原始字典。
    :return: 一个处理过的、适合用于商品推荐的字典。
    """
    processed_recommendation_data = {}

    # 原始数据包含一个顶层键，先获取其内部的方案字典
    # 例如，如果输入是 {'data': {'方案一': ...}}, 则取 'data'
    # 根据您提供的结构，顶层键是 '方案一' 的父级，我们直接用 .values() 获取
    # 但您给的例子中，'方案一'已经是顶层，所以直接遍历 raw_data

    for plan_name, plan_details in raw_data.items():
        if not isinstance(plan_details, dict):
            continue

        # 提取 '简介' 中的核心信息
        synopsis_dict = plan_details.get('简介', {})
        core_view = synopsis_dict.get('核心看点', '')
        value_promise = synopsis_dict.get('价值承诺', '')
        summary = f"核心看点: {core_view} 价值承诺: {value_promise}".strip()

        # 构建清理后的数据结构
        cleaned_plan = {
            'video_id': plan_name,  # 添加一个唯一的ID，方便引用
            '定位': plan_details.get('定位', '未知'),
            '标题': plan_details.get('标题', ''),
            '标签': plan_details.get('标签', []),
            '摘要': summary
        }

        processed_recommendation_data[plan_name] = cleaned_plan

    return processed_recommendation_data

def gen_property_good(video_info):
    """
    根据视频信息生成合适的商品信息
    """
    print(f"\n\n正在初步生成商品信息，视频信息")
    retry_delay = 10
    max_retries = 3
    format_video_info = {}
    title_schemes = video_info.get('title_schemes', [])
    titles = format_title(title_schemes)
    format_video_info['titles'] = titles

    danmu_info = video_info.get('danmu_info') or {}
    video_anlyse = danmu_info.get('视频分析', {})
    format_video_info['video_anlyse'] = video_anlyse

    comment_list = video_info.get('hudong', {}).get('comment_list', [])
    temp_comments = [(c[0], c[1]) for c in comment_list]
    # 按照c[1]降序排序，截取前100
    temp_comments = sorted(temp_comments, key=lambda x: x[1], reverse=True)[:100]

    format_video_info['comments'] = temp_comments

    prompt = f"{base_prompt}\n输入视频信息如下:\n{format_video_info}"

    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = get_llm_content(prompt=prompt, model_name="gemini-2.5-flash")
            video_info = string_to_object(raw)
            return video_info , format_video_info
        except Exception as e:
            print(f"[ERROR] 生成视频信息失败 (尝试 {attempt}/{max_retries}): {e} {raw}")
            if attempt < max_retries:
                print(f"[INFO] 正在重试... (等待 {retry_delay} 秒)")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print("[ERROR] 达到最大重试次数，失败.")
                return None, None  # 达到最大重试次数后返回 None
            traceback.print_exc()

def filter_property_good(property_goods, limit_count=80):
    """
    过滤商品信息，只保留佣金比例大于 min_commission_rate 的商品。
    """
    # 如果 property_goods 已经是一个列表，则直接使用
    if isinstance(property_goods, list):
        property_goods_list = sorted(property_goods, key=lambda x: float(x.get('commission_rate_pct', 0)),
                                     reverse=True)[:limit_count]

        return property_goods_list
    property_goods_list = []
    for key, good in property_goods.items():
        if good.get('promo_price', 0) and float(good.get('promo_price', 0)) < 100:
            property_goods_list.append(good)
    print(f"过滤商品信息，保留佣金比例大于 0 的商品，当前商品数量: {len(property_goods_list)} 原始商品数量: {len(property_goods)}")

    # 按照commissionRate降序排序，取前20个
    property_goods_list = sorted(property_goods_list, key=lambda x: float(x.get('commission_rate_pct', 0)), reverse=True)[:limit_count]
    return property_goods_list

def gen_final_property_good(video_info, property_goods):
    """
    根据视频信息生成合适的商品信息
    """
    print(f"正在生成最终商品信息，视频信息")
    property_goods_list = filter_property_good(property_goods)

    format_property_goods_list = []
    # 只保留 outerId 和 goodsName和 description和shopName
    for metadata in property_goods_list:
        format_property_goods_list.append({
            'outerId': metadata.get('outerId', ''),
            'goodsName': metadata.get('goodsName', ''),
            'brand': metadata.get('brand', 0),
            'shopName': metadata.get('shopName', ''),
            'coupon_value': metadata.get('coupon_value', 0),
            'promo_price': metadata.get('promo_price', 0),
            'leaf_category': metadata.get('leaf_category', 0),
        })

    retry_delay = 10
    max_retries = 3
    format_video_info = {}
    title_schemes = video_info.get('title_schemes', [])
    titles = format_title(title_schemes)
    format_video_info['titles'] = titles

    danmu_info = video_info.get('danmu_info') or {}
    video_anlyse = danmu_info.get('视频分析', {})
    format_video_info['video_anlyse'] = video_anlyse

    comment_list = video_info.get('hudong', {}).get('comment_list', [])
    temp_comments = [(c[0], c[1]) for c in comment_list]
    # 按照c[1]降序排序，截取前100
    temp_comments = sorted(temp_comments, key=lambda x: x[1], reverse=True)[:100]

    format_video_info['comments'] = temp_comments

    format_video_info['goods'] = format_property_goods_list

    prompt = f"{final_prompt}\n输入信息如下:\n{format_video_info}"

    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = get_llm_content(prompt=prompt)
            video_info = string_to_object(raw)
            return video_info
        except Exception as e:
            print(f"[ERROR] 生成视频信息失败 (尝试 {attempt}/{max_retries}): {e} {raw}")
            if attempt < max_retries:
                print(f"[INFO] 正在重试... (等待 {retry_delay} 秒)")
                time.sleep(retry_delay)  # 等待一段时间后再重试
            else:
                print("[ERROR] 达到最大重试次数，失败.")
                return None  # 达到最大重试次数后返回 None
            traceback.print_exc()


def extract_taokouling(text: str):
    """
    从文本中提取所有符合规则的淘口令。
    淘口令特征：
      - 以 ￥ 或 ¥ 成对包裹
      - 内部 4~64 字符，仅包含字母/数字/空格
      - 必须同时含字母和数字
    返回：
      - 匹配到的淘口令列表（格式：{'raw': 原始匹配, 'inner': 内部口令, 'span': (start, end)})
    """
    # 匹配模式：成对的￥/¥，中间不含￥/¥，长度2~64
    pattern = re.compile(r"([￥¥])\s*([^￥¥\n]{2,64}?)\s*\1")
    # 匹配零宽字符范围
    zw_chars = re.compile(r"[\u200b-\u200f\u202a-\u202e]")

    results = []
    for m in pattern.finditer(text):
        symbol = m.group(1)
        inner_raw = m.group(2)

        # 清理零宽符和多余空格
        inner = zw_chars.sub("", inner_raw)
        inner = re.sub(r"\s+", " ", inner.strip())

        # 校验规则
        if 4 <= len(inner) <= 64 \
                and re.fullmatch(r"[A-Za-z0-9 ]+", inner) \
                and any(c.isalpha() for c in inner) \
                and any(c.isdigit() for c in inner):
            results.append({
                "raw": m.group(0),
                "inner": inner,
                "span": m.span(),
                "normalized": f"{symbol}{inner}{symbol}"
            })
    return results[0]['normalized'] if results else ''


def send_good_comment(
    total_cookie,
    commenter: Any,
    bvid: str,
    final_goods_record: Dict[str, Any]
):
    """
    发送商品评论到指定的 B 站视频，并将评论置顶。

    Args:
        commenter: 带有 post_comment 和 pin_comment 方法的对象。
        bvid: 视频的 BVID 字符串。
        final_goods_record: 包含 'final_goods' 和 'property_goods' 键的字典，
                            其中 'product_recommendations' 是待推荐商品列表。
    """
    print(f"\n\n正在发送商品评论到视频 {bvid}")
    # 1. 获取并按分数降序排序推荐列表
    recommendations: List[Dict[str, Any]] = (
        final_goods_record
        .get('final_goods', {})
        .get('product_recommendations', [])
    )
    sorted_recs = sorted(
        [
            item for item in recommendations
            if (float(item.get('estimated_ctr') or 0) * float(item.get('score') or 0)) >= 0
        ],
        key=lambda item: float(item.get('estimated_ctr') or 0) * float(item.get('score') or 0),
        reverse=True
    )
    print(f"找到 {len(sorted_recs)} 条商品推荐，按预估点击率和评分排序。过滤前推荐数量: {len(recommendations)}")
    # 2. 获取完整商品信息列表
    property_goods: List[Dict[str, Any]] = final_goods_record.get('property_goods', [])

    for rec in sorted_recs:
        outer_id: str = rec.get('outerId', '')
        if not outer_id:
            continue

        # 3. 在 property_goods 找到对应商品
        target_good: Optional[Dict[str, Any]] = next(
            (pg for pg in property_goods if pg.get('outerId') == outer_id),
            None
        )
        if not target_good:
            continue
        taokouling_30d = target_good.get('taokouling_30d', '').strip()
        kouling = extract_taokouling(taokouling_30d)
        abd_image_path = target_good.get('abd_image_path', '')
        if not kouling:
            print(f"⚠️ 商品 {outer_id} 没有有效的短链接，跳过。{taokouling_30d}")
            continue
        pinned_text: str = rec.get('pinned_comment', '').strip()

        comment_body = f"{pinned_text}\n\n\n{kouling}整段内容復制，然后去 👉【🍑宝】就能直达。"

        # 4. 发布评论
        print(f"正在发布商品评论: 视频 {bvid}，商品 {outer_id} “{rec.get('goodsName', '')}” comment_body: {comment_body}")
        if os.path.exists(abd_image_path):
            rpid = commenter.post_comment(bvid=bvid, message_content=comment_body, image_path=abd_image_path)
        else:
            rpid = commenter.post_comment(bvid=bvid, message_content=comment_body)
        if not rpid:
            # 发布失败，尝试下一个
            continue

        # 5. 置顶评论并结束
        if commenter.pin_comment(bvid=bvid, rpid=rpid):
            print(f"✅ 已成功发送并置顶商品评论: 视频 {bvid}，商品 {outer_id} “{rec.get('goodsName', '')}” pinned_text: {pinned_text}")
            return rpid, rec.get('goodsName', '')

    # 如果所有推荐都处理完仍未成功
    print(f"⚠️ 未能发送或置顶任何商品评论到视频 {bvid}")
    return None, None


def add_good_comment_for_video(user_name='qiqi'):
    """
    为视频增加合适的商品链接
    """
    print(f"\n\n开始为用户 {user_name} 的视频增加商品评论...")
    config_map = init_config()
    all_records_file = f"{BASE_DIR}/{user_name}_record_info.json"
    # 找到对应的 UID
    uid = '1223805908'
    for key, value in config_map.items():
        if value['name'] == user_name:
            uid = key
            break
    # update_local_goods_info(user_name)
    total_cookie = config_map[uid]['total_cookie']
    csrf_token = config_map[uid].get('BILI_JCT', '')
    all_params = config_map[uid].get('all_params', {})
    commenter = BilibiliCommenter(total_cookie=total_cookie, csrf_token=csrf_token,all_params=all_params)
    temp_found_videos = commenter.get_user_videos(mid=uid, desired_count=10)
    metadata_cache_with_uploads = read_json('../../LLM/TikTokDownloader/metadata_cache_with_uploads.json')
    all_records = read_json(all_records_file)
    # success_bvids = read_json(success_bvids_file)
    success_bvids = [record['bvid'] for record in all_records.values() if record.get('status') == 'success' or record.get('rpid')]
    # success_bvids = []

    print(f"已处理 {len(all_records)} 条记录，其中 {len(success_bvids)} 条成功。")
    # 过滤出已经处理过的
    processed_bvids = success_bvids
    videos_to_process = [video for video in temp_found_videos if video['bvid'] not in processed_bvids]
    print(f"{user_name} 找到 {len(videos_to_process)} 个未处理的视频。总共视频数量：{len(temp_found_videos)}")

    for video in videos_to_process:
        try:
            bvid = video['bvid']
            target_value = find_video_by_bvid(bvid, metadata_cache_with_uploads) or {}
            if not target_value:
                print(f"视频 {bvid} 在 metadata_cache_with_uploads.json 中未找到对应信息，跳过。")
                continue
            record = all_records.get(bvid, {})
            if 'property_good_info' in record and record['property_good_info']:
                print(f"视频 {bvid} 已经有商品信息，跳过。")
                property_good_info = record['property_good_info']
                # format_video_info = record.get('video_info', {})
            else:
                print(f"正在处理视频 {bvid} 的商品信息...")
                property_good_info, format_video_info = gen_property_good(target_value)
            if property_good_info:
                if bvid not in all_records:
                    all_records[bvid] = {}
                all_records[bvid]['bvid'] = bvid
                all_records[bvid]['user_name'] = user_name
                all_records[bvid]['property_good_info'] = property_good_info
                # all_records[bvid]['video_info'] = format_video_info
                save_json_safe(all_records_file, all_records)

                keyword_list = [good['product_name'] for good in property_good_info['product_recommendations']]
                for good in property_good_info['product_recommendations']:
                    keyword_list.extend(good['keywords'])
                keyword_list = list(set(keyword_list))

                # if 'property_goods' in record and record['property_goods']:
                #     print(f"视频 {bvid} 已经候选商品信息，跳过。")
                #     property_goods = record['property_goods']
                # else:
                print(f"为视频 {bvid} 生成商品信息，关键词列表长度 {len(keyword_list)} 关键词列表：{keyword_list}")
                property_goods = search_goods(keyword_list)
                all_records[bvid]['property_goods'] = filter_property_good(property_goods)
                save_json_safe(all_records_file, all_records)
                if 'final_goods' in record and record['final_goods'] and True:
                    print(f"视频 {bvid} 已经有最终商品信息，跳过。")
                    final_goods = record['final_goods']
                else:
                    final_goods = gen_final_property_good(target_value, property_goods)
                    all_records[bvid]['final_goods'] = final_goods
                    save_json_safe(all_records_file, all_records)
                if final_goods:
                    rpid, good_name = send_good_comment(total_cookie, commenter, bvid, all_records[bvid])
                    if rpid:
                        all_records[bvid]['status'] = 'success'
                        all_records[bvid]['rpid'] = rpid
                        all_records[bvid]['good_name'] = good_name
                        all_records[bvid]['upload_time'] = time.time()
                        all_records[bvid]['property_goods'] = []
                        save_json_safe(all_records_file, all_records)
        except Exception as e:
            print(f"处理视频 {bvid} 时出错: {e}")
            traceback.print_exc()
            all_records[bvid]['status'] = 'error'
            all_records[bvid]['error_message'] = str(e)
            save_json_safe(all_records_file, all_records)


def update_local_goods_info(user_name='ruru'):
    """
    拉取选品车中的商品信息，并更新本地 JSON 文件。
    """
    goods_file = f"{BASE_DIR}/{user_name}_goods_info.json"
    goods_info = read_json(goods_file)
    count = 0
    car_items = list_selection_car_items(get_config(f'{user_name}_bilibili_total_cookie'), 100)
    for car_item in car_items:
        outer_id = car_item.get('outerId', '').split('-')[-1]  # 确保只取最后一部分
        if outer_id:
            car_item['updateTime'] = time.time()  # 添加更新时间
            goods_info[outer_id] = car_item
            count += 1
    save_json_safe(goods_file, goods_info)
    print(f'查询到选品车商品个数 {len(car_items)} 更新成功个数 {count}')


def worker_process_loop(user_name, interval):
    """
    这是一个长期运行的工作进程函数。
    它会为一个指定的用户重复执行任务，并确保每次任务的启动间隔至少为 interval 秒。
    """
    print(f"[进程 {multiprocessing.current_process().pid} | 用户 {user_name}] 已启动，工作周期为 {interval} 秒。")

    while True:
        start_time = time.time()

        print(
            f"[{time.strftime('%H:%M:%S')}] [进程 {multiprocessing.current_process().pid} | 用户 {user_name}] -------> 新一轮周期开始 <-------")

        try:
            # 执行核心任务
            add_good_comment_for_video(user_name)
            auto_replay(user_name)
        except Exception as e:
            # 关键：捕获任务中可能出现的任何异常，防止整个进程崩溃
            print(
                f"[{time.strftime('%H:%M:%S')}] [进程 {multiprocessing.current_process().pid} | 用户 {user_name}] 任务执行出错: {e}")

        end_time = time.time()

        # 计算任务实际花费的时间
        elapsed_time = end_time - start_time
        print(
            f"[{time.strftime('%H:%M:%S')}] [进程 {multiprocessing.current_process().pid} | 用户 {user_name}] 本轮任务耗时 {elapsed_time:.2f} 秒。")

        # 计算需要等待的时间
        wait_time = interval - elapsed_time

        if wait_time > 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] [进程 {multiprocessing.current_process().pid} | 用户 {user_name}] 等待 {wait_time:.2f} 秒后进入下一轮...")
            time.sleep(wait_time)
        else:
            # 如果任务执行时间已经超过了设定的周期，就立刻开始下一轮
            print(
                f"[{time.strftime('%H:%M:%S')}] [进程 {multiprocessing.current_process().pid} | 用户 {user_name}] 任务耗时已超出周期，立即开始下一轮。")

def auto_replay(user_name):
    """
    自动扫描进行置顶文案的回复增加购买的几率
    """
    all_records_file = f"{BASE_DIR}/{user_name}_record_info.json"
    all_records = read_json(all_records_file)
    config_map = init_config()
    commenter_map = {}
    today = datetime.date.today().isoformat()
    for key, detail_config in config_map.items():
        name = detail_config.get('name', key)
        if user_name == name:
            continue
        all_params = detail_config.get('all_params', {})
        commenter_map[name] = BilibiliCommenter(
            total_cookie=detail_config.get('total_cookie', ''),
            csrf_token=detail_config.get('BILI_JCT', ''),all_params=all_params
        )
        print(f"已创建评论者 {name} (UID: {key})")
    for bvid, record in all_records.items():
        try:
            success_count = 0
            rpid = record.get('rpid')
            good_name = record.get('good_name', '')
            exist_shill_comments = record.get('exist_shill_comments', [])
            exist_shill_users = record.get('exist_shill_users', [])
            last_processed_date = record.get('last_processed_date', '')
            if len(exist_shill_comments) >= 2:
                if not rpid or not good_name or last_processed_date == today:
                    # print(f"{user_name} 视频 {bvid} 没有 rpid，{rpid},  没有 good_name，{good_name}跳过。最近处理日期 {last_processed_date}，今天日期 {today}")
                    continue
            product_recommendations = record.get('final_goods', {}).get('product_recommendations', [])
            # 遍历product_recommendations找到good_name对应的商品
            target_product = None
            for product in product_recommendations:
                if product.get('goodsName') == good_name:
                    target_product = product
                    break
            shill_comments = []
            if target_product:
                shill_comments = target_product.get('shill_comments', [])
                # 去除已经存在的评论
                shill_comments = [comment for comment in shill_comments if comment not in exist_shill_comments]
            commenter_items = list(commenter_map.items())
            random.shuffle(commenter_items)
            for shill_comment in shill_comments:
                if success_count > 3: # 单次回复数量限制
                    print(f"用户 {user_name} 已经成功回复 {success_count} 条评论，跳过剩余评论。")
                    time.sleep(100)
                    break
                for commenter_name, commenter in commenter_items:
                    if commenter_name in exist_shill_users:
                        continue
                    reply_rpid = commenter.reply_to_comment(
                        bvid=bvid, message_content=shill_comment,
                        root_rpid=rpid, parent_rpid=rpid
                    )
                    if reply_rpid:
                        success_count += 1
                        exist_shill_comments.append(shill_comment)
                        exist_shill_users.append(commenter_name)
                        # 更新记录
                        record['exist_shill_comments'] = exist_shill_comments
                        record['exist_shill_users'] = exist_shill_users
                        print(f"{commenter_name} 回复用户 {user_name} 成功: 视频 {bvid}，评论 {reply_rpid} 内容: {shill_comment}")
                        break
                    else:
                        time.sleep(100)
        except Exception as e:
            print(f"用户 {user_name} 回复视频 {bvid} 时出错: {e}")
            traceback.print_exc()
        finally:
            record['last_processed_date'] = today
            all_records[bvid] = record
            save_json_safe(all_records_file, all_records)



if __name__ == '__main__':
    username_list = ['jun', 'tao', 'yan','nana', 'qiqi', 'jie', 'ruru', 'xue']
    username_list = ['xue']
    RUN_INTERVAL_SECONDS = 3600  # <--- 实际使用时请改为 3600

    print("--- 主进程启动，准备为每个用户创建独立的子进程 ---")

    processes = []
    # 遍历用户列表，为每个用户创建一个进程
    for user in username_list:
        p = multiprocessing.Process(target=worker_process_loop, args=(user, RUN_INTERVAL_SECONDS))
        processes.append(p)
        p.start()  # 启动进程
        print(f"已为用户 <{user}> 启动进程，PID: {p.pid}")

    # 主进程等待所有子进程结束。
    # 因为子进程是无限循环，所以主进程会一直在这里等待，直到您手动停止程序 (例如按 Ctrl+C)。
    for p in processes:
        p.join()

    print("--- 所有子进程已终止 ---")
