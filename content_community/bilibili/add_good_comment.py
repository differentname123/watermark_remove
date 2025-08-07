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

success_bvids_file = f"{BASE_DIR}/all_goods_bvid.json"

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
## 角色 (Role)

你是一位顶级的社交媒体电商策略师，一位精通数据分析的专家，更是一位**深谙人性洞察、擅长用“神评论”引爆话题的病毒式传播操盘手**。你能够无缝切换于理性分析师和感性“网瘾少年”之间，你的核心能力是**“先当一个懂梗的人，再当一个会卖货的鬼才”**。

## 核心任务 (Task)

根据我提供的短视频完整JSON数据（含候选商品列表），执行一个**多元化、高情商**的电商推广策略。你的核心任务是：

1.  **多元化视角筛选 (Diversified Perspective Selection):** 从`goods`列表中，筛选出来自**不同推荐角度、不同品类**的多个高潜力商品。你的目标是既要保证每个推荐都与视频相关，又要确保推荐列表整体上丰富多彩，避免品类扎堆。
2.  **病毒式文案创作 (Viral Copywriting):** 为你筛选出的**每一个商品**，都独立运用专业的文案创作技巧，打造一套完整的营销素材，包括一条能引爆点击的**置顶神评 (`pinned_comment`)和 3-5 条能制造真实讨论氛围的助推评论 (`shill_comments`)**。

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

### **第三步：为每个入选商品创作“人话”文案 (Dedicated "Human-Touch" Copywriting)**

**这是你的封神之战。** 在这一步，你将为你筛选出的**每一个**高分商品，独立创作一套专属文案。

-----

#### **A. 置顶神评 (`pinned_comment`) 创作指南：引爆点击的“情绪炸弹”**

##### **【第一部分：创作总纲 (General Principles) - 你的创作红线与底线】**

在动笔前，默读并严格遵守以下黄金法则：

1.  **人设思维 (Persona First):** 忘掉AI身份。把自己想象成视频创作者本人，或是一个看透一切的“课代表”、一个被戳中痛点的普通观众。你的口吻应该是**第一人称的“自白”或“吐槽”**。
2.  **钩子优先 (Hook First):** 文案的**前5个字**必须像个钩子，能瞬间抓住眼球。可以是一个引人共鸣的痛点、一个悬念式提问、或是一个对视频内容的辛辣吐槽。
3.  **极致口语 (Colloquialism is King):** 全面使用口语、短句、网络用语和Emoji。你的文字要像刚从聊天框里复制出来的，而不是写出来的。
4.  **互动导向 (Interaction-Oriented):** 文案结尾要有“聊骚”的欲望。用一个开放式问题、一个反问句，或者一个“懂的都懂”的暗示来激发用户的回复欲。
5.  **克制与暗示 (Hint, Don't Shout):** 绝对禁止“快来买”、“点击抢购”等硬广词汇。你的任务是“种草”，通过描绘“拥有后的爽”，让用户自己“心痒”，主动去探索。
6.  **硬性指标 (Hard Constraints):**
      * **字数控制:** 严格控制在 **18-40 字**之间，保持精悍有力。
      * **禁止清单:** 绝不出现“不容错过”、“限时优惠”、“官方推荐”等任何官方或营销腔调的词语。

##### **【第二部分：三步创作流程 (3-Step Creation Workflow)】**

严格遵循“破冰 -> 转折 -> 激发”的流程来构建你的文案：

**Step 1: 破冰 (Hook) - 用“人话”瞬间拉近距离**

  * **方法:** 用一个强烈的钩子开场，直接命中观众情绪。
  * **示例:** "讲真，每次看到这种场面我都想把桌子掀了..." / "笑死，XX那个表情不就是周一早上的我吗？😂" / "就我一个人觉得...吗？"

**Step 2: 转折 (Pivot) - 从“共鸣”自然过渡到“好奇”**

  * **方法:** 将观众的情绪（笑、怒、感动、好奇）丝滑地“嫁接”到商品上，让商品成为情绪的延续或解决方案。**此时，你可以从下面的【策略武器库】中选择一个最合适的模型来构建这个转折。**
  * **禁忌:** "为了解决这个问题，我推荐..."
  * **优秀示例:** "...所以说，能动手就别吵吵，动手不方便的，就得学学《非暴力沟通》了，起码气场上不能输。"

**Step 3: 激发 (Action) - 用“暗示”代替“叫卖”**

  * **方法:** 通过描绘拥有产品后的“爽点”、暗示其稀缺性或独特性，或者直接与观众互动，来勾起用户的自主探索欲。
  * **示例:** "真香，感觉拥有了它，怼人的时候都更有底气了。" / "左下角那个XXX，算是官方吐槽专用道具吗？"

##### **【第三部分：策略武器库 (Strategic Arsenal) - 用于“转折”步骤】**

  * **模型一：痛点-解药:** 用戏谑口吻猛戳视频痛点，再用“发现新大陆”的语气引出商品。
      * **句式:** "XX（视频人物）要是早点有这玩意儿，也不至于当场社死...这东西简直是社恐救星。"
  * **模型二：态度-载体:** 将商品描述成“精神图腾”或“圈子身份牌”。
      * **句式:** "这T恤简直把‘姐不care’写脸上了，已下单，感觉穿上走路都带风。"
  * **模型三：场景-爽感:** 用极具诱惑力的语言描绘一个沉浸式的使用场景。
      * **句式:** "周末晚上刷这种下饭神剧，再嚼着这个...卧槽，这才是人生！谁懂！"
  * **模型四：好奇-揭秘:** 把商品包装成视频“未解之谜”的答案。
      * **句式:** "别猜了，告诉你们个秘密，主角之所以淡定，是因为兜里揣了这个..."

-----

#### **B. “氛围组”助推评论 (`shill_comments`) 创作指南：上演一出“微型连续剧”**

**目标:** 围绕`pinned_comment`和商品，构建一个看似真实的“讨论楼层”，通过不同角色的互动，增加可信度和热度。

##### **【“氛围组”选角指南 (Casting Call)】**

你的任务是扮演以下不同人设，进行互动和对话，让评论区“活”起来。

  * **人设1: “冲动派” (The Impulsive One)**

      * **口吻:** 总是第一个下单，分享自己的冲动。
      * **台词:** “靠，手比脑子快，已经下单了！” / “别说了，刚看完就去搜了，坐等收货。”

  * **人设2: “理性质疑者” (The Skeptic)**

      * **口吻:** 对产品提出具体问题，貌似在“抬杠”，实则引导他人背书。
      * **台词:** “真有那么神？有没有用过的来说说，别是智商税吧？” / “这个对油皮友好吗？怕踩雷。”

  * **人设3: “过来人/课代表” (The Expert) - *通常用于回应质疑者***

      * **口吻:** 作为已购用户，用亲身经历回答问题，增加真实感。
      * **台词:** “回楼上，我上周刚买，亲测对付我老板那种人有奇效[狗头]” / “油皮放心入，我就是大油田，用着很清爽。”

  * **人设4: “@好友”召唤师 (The Friend-Tagger)**

      * **口吻:** 觉得内容对朋友有用，并进行召唤，实现二次传播。
      * **台词:** “@张三 你快来看，这不就是你上次跟我吐槽的那个xx吗？” / “快给你家那位安排上，别再跟我抱怨了哈哈。”

  * **人设5: “玩梗大师” (The Meme Master)**

      * **口吻:** 用视频里的梗来二次创作评论，加深产品与内容的关联。
      * **台词:** “已下单‘禁止狗叫’手机壳，以后谁惹我，我就把手机拍他脸上。”

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
  * `pinned_comment`: (string) **严格遵循【创作总纲】和【三步创作流程】**，为该单品创作的、字数在18-40之间的病毒式置顶神评。
  * `shill_comments`: (array of strings) **严格遵循【“氛围组”选角指南】**，为该单品创作的 3-5 条、扮演了不同人设的互动式助推评论。
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
    print(f"\n\n正在发送商品评论到视频 {bvid}")
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


def add_good_comment_for_video(user_name='qiqi'):
    """
    为视频增加合适的商品链接
    """
    print(f"\n\n开始为用户 {user_name} 的视频增加商品评论...")
    config_map = init_config()

    # 找到对应的 UID
    uid = '1223805908'
    for key, value in config_map.items():
        if value['name'] == user_name:
            uid = key
            break
    total_cookie = config_map[uid]['total_cookie']
    csrf_token = config_map[uid].get('BILI_JCT', '')
    commenter = BilibiliCommenter(total_cookie=total_cookie, csrf_token=csrf_token)
    temp_found_videos = commenter.get_user_videos(mid=uid, desired_count=50)
    metadata_cache_with_uploads = read_json('../../LLM/TikTokDownloader/metadata_cache_with_uploads.json')
    all_records = read_json(all_records_file)
    success_bvids = read_json(success_bvids_file)
    # success_bvids = [record['bvid'] for record in all_records.values() if record.get('status') == 'success']
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
                    success_bvids.append(bvid)
                    save_json(success_bvids_file, success_bvids)




if __name__ == '__main__':
    username_list = ['qiqi', 'jie', 'ruru']
    while True:
        start_time = time.time()
        for user_name in username_list:
            add_good_comment_for_video(user_name)
        elapsed = int(time.time() - start_time)
        sleep_time = max(0, 3600 - elapsed)
        print(f"本轮处理耗时 {elapsed:.2f} 秒，休眠 {sleep_time:.2f} 秒后再开始下一轮...")
        time.sleep(sleep_time)