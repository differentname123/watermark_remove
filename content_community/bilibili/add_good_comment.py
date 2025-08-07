import time
import traceback
from typing import Any, Dict, List, Optional

from LLM.gemini import get_llm_content
from common_utils.common_utils import read_json, get_config, save_json
from content_community.bilibili.bili_utils import fetch_goods, update_short_url
from content_community.bilibili.comment import BilibiliCommenter
from content_community.bilibili.get_comment import get_bilibili_comments
from content_community.bilibili.get_danmu import string_to_list
from content_community.bilibili.high_quality_hudong import init_config, find_video_by_bvid
from common_utils.common_utils import string_to_object, save_json, read_json

BASE_DIR = 'goods_info'

all_records_file = f"{BASE_DIR}/all_goods_info.json"


base_prompt = """
## 角色 (Role)
你是一位顶级的社交媒体电商运营专家与数据分析师。你的核心能力是深度解析视频内容与观众情绪，从而精准推荐具有高转化潜力与市场热度的商品。

## 核心任务 (Task)
根据我提供的、关于一个短视频的完整JSON数据，为该视频推荐 5-7 个最适合推广的商品。你的分析过程需要严谨，并以一个结构清晰、信息丰富的JSON格式输出最终结果。

## 输入数据说明 (Input Data)
我将提供一个JSON对象，其中包含三大关键信息模块：
1.  `titles`: 视频的多角度营销包装方案。
2.  `video_anlyse`: 对视频内容的客观结构化分析。
3.  `comments`: 真实观众的热门评论，**这是分析的核心依据**。

## 处理指令 (Processing Instructions)
1.  **深度综合分析**: 全面分析所有输入信息。洞察视频的核心要素，例如本案例中的：**饭局社交、情侣关系、朋友拆台、戏剧性尴尬（社死）、高光梗（“原文是条狗”）、以及观众情绪（对男方动机的质疑、对女方“恋爱脑”的共情或担忧）**。
2.  **商品匹配黄金原则**:
    * **强关联性 (Relevance)**: 商品必须与视频的核心场景、主题、情绪或高光梗有直接、强力的联系。
    * **大众化优先 (Popularity-First)**: 优先推荐认知度高、受众广、决策成本低的大众消费品。
    * **需求激发 (Demand Stimulation)**: 思考商品是否能满足观众被视频激发的需求（如：化解尴尬、提升情商、获得快乐等）。
3.  **提取关键词**: 在分析的基础上，为每个推荐的商品提炼出最相关的搜索关键词。
4.  **综合评分**: 在内心评估每个商品的“内容相关度”和“市场转化潜力”，然后给出一个最终的综合推荐分数。

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
````

### **JSON 字段详细解释:**

  * `product_name`: (string) **具体的商品名称或品类。** 例如: "三只松鼠坚果礼盒", "社交类桌游", "《非暴力沟通》"。**注意：这必须是一个简短的名称，而不是一个描述性长句。**
  * `reason`: (string) **简明扼要的核心推荐理由。** 必须直接关联视频内容、高光梗或观众普遍情绪。
  * `score`: (integer) **一个 1-10 的综合推荐指数。** 这个分数是你结合了**“内容相关度”**和**“市场转化潜力”**后给出的最终评分 (1=不推荐, 10=强烈推荐)。
  * `keywords`: (array of strings) **一个包含核心搜索关键词的字符串数组。** 这些关键词应高度概括商品特点并紧密结合视频热点，适合在电商平台（如淘宝、抖音商城）进行搜索。
"""


final_prompt = """
## 角色 (Role)

你是一位顶级的社交媒体电商策略师、资深数据分析师与精通营销方法论的文案专家。你的核心能力是独立思考和决策，通过**多维视角**分析内容，从给定的商品池中自主筛选出**品类丰富且均具备高转化潜力**的商品组合，并为**每一个**入选的商品量身打造一套独立的营销文案。

## 核心任务 (Task)

根据我提供的短视频完整JSON数据（含候选商品列表），执行一个**多元化**的电商推广策略。你的核心任务是：

1.  **多元化视角筛选 (Diversified Perspective Selection):** 从`goods`列表中，筛选出来自**不同推荐角度、不同品类**的多个高潜力商品。你的目标是既要保证每个推荐都与视频相关，又要确保推荐列表整体上丰富多彩，避免品类扎堆。
2.  **独立文案创作 (Independent Copywriting):** 为你筛选出的**每一个商品**，都独立运用专业的文案创作技巧，打造一套完整的营销素材，包括一条“推广文案”（用于置顶）和 3-5 条“助推评论”。

## 输入数据说明 (Input Data)

我将提供一个JSON对象，其中包含四大关键信息模块：

  * `titles`: 视频的多角度营销包装方案。
  * `video_anlyse`: 对视频内容的客观结构化分析。
  * `comments`: 真实观众的热门评论，**是洞察用户需求和情绪的核心依据**。
  * `goods`: 一个**候选商品列表**，格式为JSON数组。你的所有推荐都必须从这里产生。

## 处理指令 (Processing Instructions)

### **第一步：深度洞察 (Deep Analysis)**

全面、深入地理解所有输入信息。精准提炼出视频的**核心要素**、**戏剧冲突**、**高光热梗**，以及观众在评论中展现出的**普遍情绪**和**潜在需求**。这是你进行一切专业判断的基础。

### **第二步：多元化视角下的策略性筛选 (Strategic Filtering from a Diversified Perspective)**

这是最关键的步骤，请严格遵循以下两阶段流程：

**2.1: 挖掘多维推荐角度 (Uncover Multi-Dimensional Recommendation Angles)**
在动筛选作之前，先强迫自己从多个不同的视角分析视频，并识别出 2-4 个**完全不同**的推荐切入点。例如：

  * **核心主题角度 (Core Theme):** 直接解决视频核心矛盾的商品。对于职场视频，这就是“职场技能/沟通类书籍”。
  * **情绪/氛围角度 (Emotion/Atmosphere):** 满足观众在观看时产生的情绪需求的商品。例如，视频很搞笑，可以推荐“零食”来“边吃边看”；视频很紧张，可以推荐“解压玩具”；视频中的人物态度很酷，可以推荐“同款态度T恤”。
  * **符号/金句角度 (Symbol/Catchphrase):** 将视频中的高光台词、梗或标志性物品实体化的商品。例如，如果角色说了“禁止狗叫”，那么印有这句话的T恤或手机壳就是绝佳选择。
  * **场景延伸角度 (Scenario Extension):** 从视频场景延伸出去的相关商品。例如，面试场景可以延伸到“提升形象的配饰”、“保持精力的咖啡”等。

**2.2: “赛道选马”式筛选 (Select the "Best-in-Class" for Each Angle)**
在确定了多个不同的推荐“角度”后，执行以下筛选原则：

  * **绝不扎堆：** **绝对不要**在同一个“角度”或“品类”下进行重复推荐。例如，一旦你从“核心主题角度”选择了一本最合适的书，就不要再选择第二本。
  * **优中选优：** 你的目标是从你挖掘出的**每一个不同“角度”中，分别挑选出那一个最匹配、得分最高的商品**。这样能确保你的最终推荐列表既有深度（强相关），又有广度（多样性）。
  * **保留高标准：** 即使某个角度有候选商品，但如果其匹配度不高（例如，综合评分低于7分），你也可以果断放弃该角度的推荐。质量永远优先于数量。

### **第三步：为每个入选商品创作专属文案 (Dedicated Copywriting for Each Selected Product)**

**针对你在上一步筛选出的每一个高分商品，分别独立执行以下文案创作流程：**

#### **A. 推广文案 (Promotional Copy) 创作指南**

  * **核心要求：** 文案必须**自成一体、文意完整**，不能依赖或指向任何外部UI元素。文案本身就要能激发强烈的点击和购买欲望。在动笔前，请先思考并选择一个最适合当前商品和视频内容的**核心文案模型**来构建你的文案。
  * **核心文案创作模型 (Core Copywriting Models):**
      * **模型一：痛点-解药 (Pain Point - Antidote)**
          * **方法：** 精准识别并放大视频中人物的尴尬、困境，或观众评论中流露的普遍焦虑（痛点）。然后，将商品作为解决这个痛点的完美方案或“解药”戏剧化地呈现出来。
          * **适用：** 功能性产品、知识付费、书籍、能解决特定问题的工具。
      * **模型二：态度-载体 (Attitude - Vehicle)**
          * **方法：** 捕捉视频传递出的核心“态度”或“情绪”（例如：搞笑、不屑、潮流、自嘲）。将商品定位为用户表达这种态度的最佳“载体”或“身份标签”，让用户通过购买来“站队”或“彰显个性”。
          * **适用：** 服饰、饰品、文创周边、具有设计感或象征意义的商品。
      * **模型三：场景-爽感 (Scenario - Pleasure)**
          * **方法：** 描绘一个用户极易代入的具体使用“场景”（例如：深夜追剧、朋友聚会、办公室下午茶）。然后用极具感染力的语言，去渲染和放大在该场景下使用该商品所带来的感官愉悦或情绪满足（即“爽感”）。
          * **适用：** 零食、饮品、美妆、香氛、解压玩具等体验型商品。
      * **模型四：好奇-揭秘 (Curiosity - Reveal)**
          * **方法：** 用一个悬念或一个引人好奇的问题开场，这个问题通常源于视频内容。然后暗示或明示，答案/秘密就藏在这个商品里，驱动用户因好奇而产生了解和购买的欲望。
          * **适用：** 内容型产品、有“黑科技”或独特卖点的商品、盲盒等。

#### **B. “氛围组”助推评论创作指南 (Shill Comments Guidelines)**

  * **目标：** 围绕**当前推荐单品**，制造“很多人在讨论”的氛围，从侧面为商品背书。
  * **风格：** 模拟不同角色的真实用户，口吻必须自然、真实、多样化。
  * **创作核心（反面教材 vs. 正面示例）：**
      * **禁忌：** “这个XXX真好用，大家快来买啊！”
      * **优秀示例 (围绕书籍):** “刚下单了那本《非暴力沟通》，希望我跟npy别再有这种尴尬了”、“有没有看过的姐妹说说，这本书对付我老板那种人管用不？”
      * **优秀示例 (围绕零食):** “笑不活了，我宣布这个鸭舌是本次视频最佳配角，已加购”、“看饿了，追剧的时候来一包这个，简直是神仙日子。”

## 输出格式要求 (Output Format Requirement)

你的回答**必须是唯一且纯粹的JSON对象**。禁止在JSON代码块前后添加任何Markdown标记、介绍、总结或任何形式的解释性文字。输出的根节点必须是一个名为 `product_recommendations` 的数组。**此数组可以包含任意数量的对象（0个、1个或多个），具体数量由你的专业判断和多元化筛选结果决定。**

```json
{
  "product_recommendations": [
    {
      "outerId": "string",
      "goodsName": "string",
      "reason": "string",
      "score": "integer",
      "keywords": ["string"],
      "pinned_comment": "string",
      "shill_comments": [
        "string",
        "string",
        "string"
      ]
    }
  ]
}
```

### **JSON 字段详细解释:**

  * `outerId`: (string) **原样返回**所选商品的 `outerId`。
  * `goodsName`: (string) **原样返回**所选商品的 `goodsName`。
  * `reason`: (string) **一句话核心推荐理由**，精炼阐述为何这个**单品**值得推荐。
  * `score`: (integer) **1-10 的综合推荐指数**，这是你做出筛选决策的关键依据。建议按分数从高到低排序。
  * `keywords`: (array of strings) **针对该单品的核心搜索关键词数组**。
  * `pinned_comment`: (string) **专门为该单品创作**的、运用了核心方法论的推广文案。
  * `shill_comments`: (array of strings) **专门为该单品创作**的 3-5 条“氛围组”助推评论。

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
            goods = fetch_goods(get_config(f'{user_name}_bilibili_total_cookie'), 20, key_word)
            # print(f"通过接口 关键词 '{key_word}' 抓取到 {len(goods)} 条商品信息。")
            # 更新local_file_path的数据
            if goods:
                local_good_data = read_json(local_file_path)
                for good in goods:
                    good['updateTime'] = current_time
                    local_good_data[good['outerId']] = good
                save_json(local_file_path, local_good_data)

        property_goods.extend(search_local_goods_info(key_word, local_file_path))
    # 去重
    property_goods = {good['outerId']: good for good in property_goods}.values()

    # 打印详细的总体信息
    print(f"抓取到 {len(property_goods)} 条商品信息。key_word_list {len(key_word_list)} 关键词列表 {key_word_list} ")
    return {good['outerId']: good for good in property_goods}


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

    video_anlyse = video_info.get('danmu_info', {}).get('视频分析', {})
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
            raw = get_llm_content(prompt=prompt)
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

def filter_property_good(property_goods, limit_count=40):
    """
    过滤商品信息，只保留佣金比例大于 min_commission_rate 的商品。
    """
    property_goods_list = []
    for key, good in property_goods.items():
        if good.get('commissionRate', 0) and float(good.get('commissionRate', 0)) > 0:
            property_goods_list.append(good)

    # 按照commissionRate降序排序，取前20个
    property_goods_list = sorted(property_goods_list, key=lambda x: float(x.get('commissionRate', 0)), reverse=True)[:limit_count]
    return property_goods_list

def gen_final_property_good(video_info, property_goods):
    """
    根据视频信息生成合适的商品信息
    """
    print(f"正在生成最终商品信息，视频信息")
    property_goods_list = filter_property_good(property_goods)

    format_property_goods_list = []
    # 只保留 outerId 和 goodsName和 description和shopName
    for good in property_goods_list:
        format_property_goods_list.append({
            'outerId': good.get('outerId', ''),
            'goodsName': good.get('goodsName', ''),
            'description': good.get('description', ''),
            'shopName': good.get('shopName', '')
        })

    retry_delay = 10
    max_retries = 3
    format_video_info = {}
    title_schemes = video_info.get('title_schemes', [])
    titles = format_title(title_schemes)
    format_video_info['titles'] = titles

    video_anlyse = video_info.get('danmu_info', {}).get('视频分析', {})
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
    # 1. 获取并按分数降序排序推荐列表
    recommendations: List[Dict[str, Any]] = (
        final_goods_record
        .get('final_goods', {})
        .get('product_recommendations', [])
    )
    sorted_recs = sorted(
        recommendations,
        key=lambda item: item.get('score', 0),
        reverse=True
    )

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

        short_url: Optional[str] = target_good.get('shortUrl')
        pinned_text: str = rec.get('pinned_comment', '').strip()
        if not short_url:
            new_target_good = update_short_url(total_cookie, [target_good])
            short_url: Optional[str] = new_target_good[0].get('shortUrl')
        if not short_url:
            print(f"⚠️ 商品 {outer_id} “{rec.get('goodsName', '')}” 缺少 shortUrl，跳过。")
            continue

        comment_body = f"{short_url}\n{pinned_text}"

        # 4. 发布评论
        rpid = commenter.post_comment(bvid=bvid, message_content=comment_body)
        if not rpid:
            # 发布失败，尝试下一个
            continue

        # 5. 置顶评论并结束
        if commenter.pin_comment(bvid=bvid, rpid=rpid):
            print(f"✅ 已成功发送并置顶商品评论: 视频 {bvid}，商品 {outer_id} “{rec.get('goodsName', '')}” pinned_text: {pinned_text}")
            return rpid

    # 如果所有推荐都处理完仍未成功
    print(f"⚠️ 未能发送或置顶任何商品评论到视频 {bvid}")
    return None


def add_good_comment_for_video(user_name='ruru'):
    """
    为视频增加合适的商品链接
    """
    config_map = init_config()
    total_cookie = config_map['1223805908']['total_cookie']
    csrf_token = config_map['1223805908'].get('BILI_JCT', '')
    # 找到对应的 UID
    uid = '1223805908'
    for key, value in config_map.items():
        if value['name'] == user_name:
            uid = key
            break
    commenter = BilibiliCommenter(total_cookie=total_cookie, csrf_token=csrf_token)
    temp_found_videos = commenter.get_user_videos(mid=uid, desired_count=50)
    metadata_cache_with_uploads = read_json('../../LLM/TikTokDownloader/metadata_cache_with_uploads.json')
    all_records = read_json(all_records_file)
    success_bvids = [record['bvid'] for record in all_records.values() if record.get('status') == 'success']
    print(f"已处理 {len(all_records)} 条记录，其中 {len(success_bvids)} 条成功。")
    # 过滤出已经处理过的
    processed_bvids = success_bvids
    videos_to_process = [video for video in temp_found_videos if video['bvid'] not in processed_bvids]
    print(f"找到 {len(videos_to_process)} 个未处理的视频。总共视频数量：{len(temp_found_videos)}")

    for video in videos_to_process:
        bvid = video['bvid']
        target_value = find_video_by_bvid(bvid, metadata_cache_with_uploads) or {}
        if not target_value:
            print(f"视频 {bvid} 在 metadata_cache_with_uploads.json 中未找到对应信息，跳过。")
            continue
        record = all_records.get(bvid, {})
        if 'property_good_info' in record and record['property_good_info']:
            print(f"视频 {bvid} 已经有商品信息，跳过。")
            property_good_info = record['property_good_info']
            format_video_info = record.get('video_info', {})
        else:
            print(f"正在处理视频 {bvid} 的商品信息...")
            property_good_info, format_video_info = gen_property_good(target_value)
        if property_good_info:
            if bvid not in all_records:
                all_records[bvid] = {}
            all_records[bvid]['bvid'] = bvid
            all_records[bvid]['status'] = 'start'
            all_records[bvid]['user_name'] = user_name
            all_records[bvid]['property_good_info'] = property_good_info
            all_records[bvid]['video_info'] = format_video_info
            save_json(all_records_file, all_records)

            keyword_list = [good['product_name'] for good in property_good_info['product_recommendations']]
            for good in property_good_info['product_recommendations']:
                keyword_list.extend(good['keywords'])
            keyword_list = list(set(keyword_list))

            if 'property_goods' in record and record['property_goods']:
                print(f"视频 {bvid} 已经候选商品信息，跳过。")
                property_goods = record['property_goods']
            else:
                print(f"为视频 {bvid} 生成商品信息，关键词列表长度 {len(keyword_list)} 关键词列表：{keyword_list}")
                property_goods = search_goods_info(keyword_list, user_name)
                all_records[bvid]['property_goods'] = filter_property_good(property_goods)
                save_json(all_records_file, all_records)
            if 'final_goods' in record and record['final_goods']:
                print(f"视频 {bvid} 已经有最终商品信息，跳过。")
                final_goods = record['final_goods']
            else:
                final_goods = gen_final_property_good(target_value, property_goods)
                all_records[bvid]['final_goods'] = final_goods
                save_json(all_records_file, all_records)
            if final_goods:
                rpid = send_good_comment(total_cookie, commenter, bvid, all_records[bvid])
                if rpid:
                    all_records[bvid]['status'] = 'success'
                    all_records[bvid]['rpid'] = rpid
                    save_json(all_records_file, all_records)




if __name__ == '__main__':
    # 示例关键词列表
    add_good_comment_for_video()
    # 可以添加更多关键词或修改关键词列表进行测试