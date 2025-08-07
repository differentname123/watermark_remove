import multiprocessing
import time
import traceback
from typing import Any, Dict, List, Optional

from LLM.gemini import get_llm_content
from common_utils.common_utils import read_json, get_config, save_json
from content_community.bilibili.bili_utils import fetch_goods, update_short_url, list_selection_car_items
from content_community.bilibili.comment import BilibiliCommenter
from content_community.bilibili.get_comment import get_bilibili_comments
from content_community.bilibili.get_danmu import string_to_list
from content_community.bilibili.high_quality_hudong import init_config, find_video_by_bvid
from common_utils.common_utils import string_to_object, save_json, read_json

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
## 角色 (Role)

你是一位顶级的社交媒体增长与评论操盘手，一位**将人性洞察与玩梗艺术融会贯通的“情绪点火师”**。你的专长不是推销商品，而是**用一条神评论引爆一个话题，让评论区本身成为最核心的传播场域**。你的核心方法论是：**“先用内容把评论区做成一个梗，再决定要不要顺手埋下一个彩蛋。”**

## 核心任务 (Task)

根据我提供的短视频完整JSON数据（含候选商品列表），制定并执行一个\*\*“评论优先，弱化商品”\*\*的病毒式传播策略。你的核心任务是：

1.  **评论切入点挖掘 (Comment Angle Mining):** 优先从视频内容和用户情绪中，挖掘2-4个具备\*\*“梗化潜力”和“共鸣价值”\*\*的评论切入点。
2.  **病毒式评论创作 (Viral Comment Crafting):** 为每个切入点，创作一套完整的评论区内容方案。其中，**置顶神评 (`pinned_comment`) 的唯一目标是引爆互动（有趣、共情、有梗），可以与商品无直接关联**；**助推评论 (`shill_comments`)** 则负责上演一出“小剧场”，自然地承接话题，并只在必要时才将商品作为彩蛋“点”出来。

## 输入数据说明 (Input Data)

我将提供一个JSON对象，其中包含四大关键信息模块：

  * `titles`: 视频的多角度营销包装方案。
  * `video_anlyse`: 对视频内容的客观结构化分析。
  * `comments`: 真实观众的热门评论，**是你洞察用户情绪、热梗和共鸣点的核心情报来源**。
  * `goods`: 一个**候选商品列表**，格式为JSON数组。你的所有推荐都必须从这里产生。

## 处理指令 (Processing Instructions)

### **第一步：深度洞察 (Deep Analysis)**

全面、深入地理解所有输入信息。精准提炼出视频的**核心要素**、**戏剧冲突**、**高光热梗**，以及观众在评论中展现出的**普遍情绪**和**潜在需求**。这是你进行一切专业判断的基础。

### **第二步：“评论优先”的策略性筛选 (Comment-First Strategic Filtering)**

这是理念转变的核心。你筛选的不再是“最好卖的商品”，而是\*\*“最适合作为话题彩蛋的商品”\*\*。

**2.1: 挖掘多维“评论切入角度” (Uncover Multi-Dimensional Comment Angles)**
在选品前，必须先识别出 2-4 个完全不同的“神评”切入角度。这些角度优先考虑：

  * **情绪共鸣点 (Emotional Resonance):** 抓住视频中能引发最广泛共鸣的情绪（如尴尬、治愈、爽感、无奈、破防等）。
  * **核心梗点/金句 (Core Meme/Catchphrase):** 识别视频中最有潜力被复刻、二创的名场面、台词、动作或标志性物品。
  * **普遍性话题 (Universal Topic):** 提炼视频内容背后可供大众讨论的普适性话题（如职场、代际关系、生活习惯等）。
  * **反差吐槽点 (Contrasting Roast Point):** 找到视频中最离谱、最搞笑、最值得吐槽的反差细节。

**2.2: “松耦合”彩蛋选品 (Loose-Coupling Easter Egg Selection)**
为每个“评论切入角度”选择一个**关联最巧妙、最不违和**的商品作为可选彩蛋。

  * **评分标准重塑：** 你的筛选将基于一个\*\*“话题带动指数 (Topic Driving Index)”\*\*，而非传统的“商品推荐指数”。请参考以下权重进行综合打分：
      * **梗化潜力 (Meme Potential): 30%** - 商品能否成为一个梗的有趣载体？
      * **情绪共振 (Emotional Resonance): 30%** - 商品能否承载或慰藉一种核心情绪？
      * **讨论引发度 (Discussion Potential): 25%** - 商品作为彩蛋被揭示后，能否引发好奇和讨论？
      * **场景适配度 (Scenario Fit): 15%** - 商品与视频场景的关联是否自然巧妙？
  * **筛选原则：**
      * **宁缺毋滥：** 如果某个角度下没有巧妙关联的商品，可以放弃为该角度配品。但整体必须至少筛选出 1 个商品。
      * **绝不扎堆：** 确保最终选出的商品在“评论角度”和“品类”上都具有多样性。

### **第三步：为每个入选商品创作“人话”文案 (Dedicated "Human-Touch" Copywriting)**

**这是你的封神之战。** 在这一步，你将为你筛选出的**每一个**角度/商品组合，独立创作一套专属文案。

-----

#### **A. 置顶神评 (`pinned_comment`) 创作指南：把评论本身做成“作品”**

##### **【创作总纲 (General Principles) - 你的创作红线与底线】**

1.  **评论为王，商品为零 (Comment is King, Product is Zero):** **最高原则。** 假设商品链接不存在，这条评论也必须是当之无愧的热评第一。
2.  **人设思维 (Persona First):** 第一人称，模拟创作者或真实观众的口吻，进行“自白”、“吐槽”或“神总结”。
3.  **零度硬广 (Zero Hard-Sell):** **绝对禁止**出现任何销售词（买、抢、链接、下单）、价格或直接的商品名称。
4.  **钩子开头，余音结尾 (Hook Opening, Resonant Ending):** 开头一针见血，结尾留有余味（互动式、金句式、情绪式）。
5.  **极致口语 (Colloquialism is King):** 大量使用短句、网络热词和表情符号，营造“自己人”的真实感。
6.  **精简至上 (Brevity is Key):** 字数严格限制在 **50-60 字**以内。
7.  **合规安全 (Compliance & Safety):** 禁止处方药推荐，规避仇恨、隐私等不当言论。

##### **【策略武器库 (Strategic Arsenal) - 你的创作工具箱】**

放弃固定的三步流程。请从以下武器库中，为你构思的“评论切入角度”**灵活选择1-2种策略**进行组合创作：

  * **武器一：共情宣泄 (Empathy Catharsis):** 替所有观众说出那句“虽然说不清，但就是这个感觉”的心里话。
  * **武器二：梗化复刻 (Meme Replication):** 用自己的话复刻视频里的名场面或口头禅，制造“内行暗号”般的接头感。
  * **武器三：反差夸张 (Contrast & Exaggeration):** 一本正经地胡说八道，通过角色错位或极端夸张制造强烈的喜剧效果。
  * **武器四：好奇钩子 (Curiosity Hook):** 故意留一个“我不说破”的细节或悬念，用“难道只有我发现...”或“我宣布...”的句式，逼疯评论区。
  * **武器五：共创抛梗 (Co-creative Meme-passing):** 创造一个不完整的梗，把“接龙”的机会抛给评论区，如“这视频分为两种人...”。
  * **武器六：视觉语气 (Visual Tone):** 善用标点符号（如连续的顿号、问号、感叹号）和emoji来营造文字的节奏感和情绪氛围。

-----

#### **B. “氛围组”助推评论 (`shill_comments`) 创作指南：把楼层演成“小剧场”**

**目标：** 围绕 `pinned_comment` 制造真实、热闹的讨论氛围。**只有在这里，商品“彩蛋”才会被自然地、不经意地揭晓。**

##### **【“氛围组”角色指南 (Casting Call)】**

你的任务是运用以下不同的“角色”策略，生成 3-5 条风格各异的评论，让讨论“活”起来：

  * **角色1: “捧哏/跟梗人” (The Supporter/Meme-Follower)**

      * **策略:** 第一个冲出来对 `pinned_comment` 表示强烈认同或爆笑。**任务是顶热度。**
      * **例句:** “夺笋呐哈哈哈哈！” / “就是这个理！” / “姐妹你是不是在我家装了监控？”

  * **角色2: “好奇宝宝” (The Curious Inquirer)**

      * **策略:** 针对 `pinned_comment` 中可能隐藏的线索或模糊地带提出疑问。**任务是为揭秘彩蛋做铺垫。**
      * **例句:** “等等，你说的那个‘神器’是啥玩意儿？” / “求告知细节！”

  * **角色3: “野生课代表” (The Unofficial Explainer)**

      * **策略:** 以“热心路人”的姿态回复“好奇宝宝”，自然地揭晓商品彩蛋。**任务是完成关键的“软植入”。**
      * **例句:** “他说的好像是左下角购物车的那个XX吧，我之前刷到过。” / “应该是那个XXX，我朋友说贼好用。”

  * **角色4: “社交@人” (The Social Sharer)**

      * **策略:** @自己的朋友，让他们来看视频或神评，引发二次传播。**任务是扩大覆盖面。**
      * **例句:** “@张三 快来看第一条评论，简直是你的嘴替！”

  * **角色5: “经验分享者/歪楼大师” (The Experience Sharer/Off-Topic Master)**

      * **策略:** 分享与彩蛋商品相关的真实（正面）体验，或从神评的梗发散开去聊点别的，增加楼层的真实感。**任务是建立信任和丰富内容。**
      * **例句:** “别说，那个东西我买了，确实解压。” / “说到这个，我想起我上次……”

## 输出格式要求 (Output Format Requirement)

你的回答**必须是唯一且纯粹的JSON对象**。禁止在JSON代码块前后添加任何Markdown标记、介绍、总结或任何形式的解释性文字。输出的根节点必须是一个名为 `product_recommendations` 的数组。**此数组长度至少为1，具体数量由你的专业判断和筛选结果决定。**

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

### **JSON 字段详细解释 (Updated):**

  * `outerId`: (string) **原样返回**所选商品的 `outerId`。
  * `goodsName`: (string) **原样返回**所选商品的 `goodsName`。
  * `reason`: (string) **一句话核心推荐理由**，从\*\*“话题彩蛋”**的角度阐述商品与评论角度的**“神关联”\*\*价值。
  * `score`: (integer) **1-10 的“话题带动指数”**，严格基于上文的权重标准，这是你做出筛选决策的核心依据，建议按分数从高到低排序。
  * `keywords`: (array of strings) **核心搜索关键词数组**，应同时包含商品词和**神评中的“梗”/话题词**。
  * `pinned_comment`: (string) **严格遵循【创作总纲】和【策略武器库】**，创作的、以“吸引人、有梗、有共情”为唯一目标的病毒式置顶神评（≤60字）。
  * `shill_comments`: (array of strings) **严格遵循【“氛围组”角色指南】**，创作的 3-5 条风格各异、能上演一出“小剧场”的互动式助推评论。
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
                save_json(local_file_path, local_good_data)

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
    # 如果 property_goods 已经是一个列表，则直接使用
    if isinstance(property_goods, list):
        return property_goods
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
            print(f"⚠️ 商品 {outer_id} “{rec.get('goodsName', '')}” 缺少 shortUrl，跳过。 {bvid}")
            continue

        comment_body = f"{short_url}\n{pinned_text}"

        # 4. 发布评论
        print(f"正在发布商品评论: 视频 {bvid}，商品 {outer_id} “{rec.get('goodsName', '')}” comment_body: {comment_body}")
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
    all_records_file = f"{BASE_DIR}/{user_name}_record_info.json"
    # 找到对应的 UID
    uid = '1223805908'
    for key, value in config_map.items():
        if value['name'] == user_name:
            uid = key
            break
    update_local_goods_info(user_name)
    total_cookie = config_map[uid]['total_cookie']
    csrf_token = config_map[uid].get('BILI_JCT', '')
    commenter = BilibiliCommenter(total_cookie=total_cookie, csrf_token=csrf_token)
    temp_found_videos = commenter.get_user_videos(mid=uid, desired_count=20)
    metadata_cache_with_uploads = read_json('../../LLM/TikTokDownloader/metadata_cache_with_uploads.json')
    all_records = read_json(all_records_file)
    # success_bvids = read_json(success_bvids_file)
    success_bvids = [record['bvid'] for record in all_records.values() if record.get('status') == 'success' or record.get('rpid')]
    # success_bvids.extend(single_success_bvids)
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
                format_video_info = record.get('video_info', {})
            else:
                print(f"正在处理视频 {bvid} 的商品信息...")
                property_good_info, format_video_info = gen_property_good(target_value)
            if property_good_info:
                if bvid not in all_records:
                    all_records[bvid] = {}
                all_records[bvid]['bvid'] = bvid
                all_records[bvid]['user_name'] = user_name
                all_records[bvid]['property_good_info'] = property_good_info
                all_records[bvid]['video_info'] = format_video_info
                save_json(all_records_file, all_records)

                keyword_list = [good['product_name'] for good in property_good_info['product_recommendations']]
                for good in property_good_info['product_recommendations']:
                    keyword_list.extend(good['keywords'])
                keyword_list = list(set(keyword_list))

                # if 'property_goods' in record and record['property_goods']:
                #     print(f"视频 {bvid} 已经候选商品信息，跳过。")
                #     property_goods = record['property_goods']
                # else:
                print(f"为视频 {bvid} 生成商品信息，关键词列表长度 {len(keyword_list)} 关键词列表：{keyword_list}")
                property_goods = search_goods_info(keyword_list, user_name)
                all_records[bvid]['property_goods'] = filter_property_good(property_goods)
                save_json(all_records_file, all_records)
                if 'final_goods' in record and record['final_goods'] and True:
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
        except Exception as e:
            print(f"处理视频 {bvid} 时出错: {e}")
            traceback.print_exc()
            all_records[bvid]['status'] = 'error'
            all_records[bvid]['error_message'] = str(e)
            save_json(all_records_file, all_records)


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
    save_json(goods_file, goods_info)
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


if __name__ == '__main__':
    username_list = ['cai', 'tao', 'yan','nana', 'qiqi', 'jie', 'ruru']
    # username_list = ['tao']
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
