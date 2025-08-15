import datetime
import glob
import multiprocessing
import os
import pathlib
import random
import time
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd

from LLM.gemini import get_llm_content
from common_utils.common_utils import read_json, get_config, save_json_safe, init_config, find_key_values, \
    process_product_title, download_public_image
from content_community.bilibili.bili_utils import fetch_goods, update_short_url, list_selection_car_items
from content_community.bilibili.comment import BilibiliCommenter
from content_community.bilibili.get_comment import get_bilibili_comments
from content_community.bilibili.get_danmu import string_to_list
from content_community.bilibili.high_quality_hudong import find_video_by_bvid
from common_utils.common_utils import string_to_object, save_json_safe, read_json
from content_community.taobao.taobao_utils import fetch_alimama_data, creat_and_favorite
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

BASE_DIR = 'goods_info'

def init_model_and_db(
    model_name="BAAI/bge-base-zh-v1.5",
    db_path="./product_db",
    collection_name="my_products",
    device="cpu",
    proxy=None
):
    """初始化模型与数据库集合"""
    if proxy:
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy

    print(f"正在加载语义模型: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    print("模型加载完成。")

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    # 清除系统代理，防止影响其它请求
    if proxy:
        del os.environ['HTTP_PROXY']
        del os.environ['HTTPS_PROXY']
    return model, collection


def add_products_from_csv(csv_file_path, model, collection, price_field="promo_price"):
    """从CSV文件添加商品到向量数据库"""
    print(f"\n--- 开始处理CSV文件: {csv_file_path} ---")
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 '{csv_file_path}' 不存在。")
        return

    try:
        df = pd.read_csv(csv_file_path)
        df.fillna('', inplace=True)
    except Exception as e:
        print(f"错误: 读取或解析CSV文件失败: {e}")
        return

    required_columns = ['outerId', 'all_str', 'goodsName']
    if not all(col in df.columns for col in required_columns):
        print(f"错误: CSV文件缺少必要的列。需要包含: {required_columns}")
        return

    ids_to_add, documents_to_add, metadatas_to_add = [], [], []
    existing_ids = set(collection.get(include=[])['ids'])
    print(f"数据库中已有 {len(existing_ids)} 个商品。")

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="处理CSV行"):
        product_id = str(row['outerId'])
        if product_id in existing_ids:
            continue

        document = str(row['all_str'])
        metadata = {
            key: (float(value) if key == price_field and isinstance(value, (int, float)) else str(value))
            for key, value in row.items()
            if key != "all_str"
        }

        ids_to_add.append(product_id)
        documents_to_add.append(document)
        metadatas_to_add.append(metadata)

    if not ids_to_add:
        print("没有新的商品需要添加。")
        return

    print(f"发现 {len(ids_to_add)} 个新商品，正在生成向量并存入数据库...")
    embeddings = model.encode(
        documents_to_add,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    collection.add(
        embeddings=embeddings.tolist(),
        documents=documents_to_add,
        metadatas=metadatas_to_add,
        ids=ids_to_add
    )
    print(f"成功添加 {len(ids_to_add)} 个新商品！当前总数: {collection.count()}")


def add_to_favorites_batch():
    """
    批量将商品加入收藏（保持原有逻辑不变）：
    - 从 all_goods_info.json 加载商品
    - 计算 unique、score（calTkRate * calTkCommission）
    - 按 unique 去重、筛选 score > 200、tkTotalSales > 10、finalPromotionPrice < 100
    - 排序（score 降序），按最多 200 个分批创建收藏夹并添加商品
    - 已存在于 all_favorites 的 unique 会被跳过
    """
    all_goods_file = f"{BASE_DIR}/all_goods_info.json"
    all_favorites_file = f"{BASE_DIR}/all_favorites_info.json"

    # 读取已有收藏（可能来自 list 或其他，可安全转为 list）
    all_favorites = read_json(all_favorites_file)
    all_favorites = list(all_favorites)

    # 读取所有商品（可能为 dict），统一为 list
    all_goods = read_json(all_goods_file)
    goods_iterable = all_goods.values() if isinstance(all_goods, dict) else all_goods
    goods: List[Dict[str, Any]] = list(goods_iterable)

    print(f"开始处理商品信息，共有 {len(goods)} 条商品信息。已有收藏夹商品 {len(all_favorites)} 条。")

    # 计算 unique 和 score（保留你的逻辑与阈值）
    for good in goods:
        itemName = process_product_title(good.get('itemName', ""))
        calTkRate = good.get('calTkRate', 0)
        calTkCommission = good.get('calTkCommission', 0)

        # 注意避免 f-string 引号冲突（保持原来 unique 格式）
        shop_title = good.get('shopTitle', "")
        final_price = good.get('finalPromotionPrice', "")
        good['unique'] = f"{shop_title}-{itemName}-{final_price}"

        # 计算 score，保持原有 behavior（异常时设为 0）
        try:
            if calTkRate and calTkCommission:
                good['score'] = float(calTkRate) * float(calTkCommission)
            else:
                good['score'] = 0
        except (ValueError, TypeError):
            good['score'] = 0

    # 按 unique 去重（保留首次出现的 entry）
    unique_goods: Dict[str, Dict[str, Any]] = {}
    for good in goods:
        unique_key = good.get('unique')
        if unique_key and unique_key not in unique_goods:
            unique_goods[unique_key] = good
    goods = list(unique_goods.values())

    # 保留 score > 200
    goods = [g for g in goods if g.get('score', 0) > 1000]

    # 过滤掉 unique 已存在于 all_favorites 的商品
    goods = [g for g in goods if g.get('unique') not in all_favorites]

    # 过滤 tkTotalSales 非空且 >10，并且 finalPromotionPrice < 100
    def passes_sales_and_price(g: Dict[str, Any]) -> bool:
        try:
            tk_sales = g.get('tkTotalSales')
            if not tk_sales:
                return False
            if float(tk_sales) <= 10:
                return False
            final_price_val = float(g.get('finalPromotionPrice', 0))
            if final_price_val >= 100:
                return False
            return True
        except (ValueError, TypeError):
            return False

    goods = [g for g in goods if passes_sales_and_price(g)]

    # 按 score 降序排列（score 已为 float 或可转 float）
    goods = sorted(goods, key=lambda x: float(x.get('score', 0)), reverse=False)

    print(f"过滤后商品信息，共有 {len(goods)} 条商品信息。")

    # 分批处理，每批最多 batch_size 个
    batch_size = 200
    batches = [goods[i:i + batch_size] for i in range(0, len(goods), batch_size)]
    batch_number = 0
    current_human_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for batch in batches:
        batch_number += 1
        title = f"{current_human_time}_{batch_number}"
        item_ids = [good['itemId'] for good in batch]
        unique_list = [good['unique'] for good in batch]

        result = creat_and_favorite(title, item_ids)
        if result:
            all_favorites.extend(unique_list)
            save_json_safe(all_favorites_file, all_favorites)
            print(f"收藏夹创建成功，标题: {title}，商品数量: {len(batch)}，批次号: {batch_number}/{len(batches)}")
        else:
            print("收藏夹创建或商品添加失败。请检查日志以获取更多信息。")
            break

def get_goods_info():
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    cookie_string_list = [get_config("jie_taobao_cookie"), get_config("zhu_taobao_cookie"), get_config("dahao_taobao_cookie")]
    user_name_list = ['cai', 'tao', 'yan', 'nana', 'qiqi', 'jie', 'ruru', 'xue']
    all_goods_file = f"{BASE_DIR}/all_goods_info.json"
    processed_keyword_file = f"{BASE_DIR}/all_processed_keywords.json"
    processed_keywords = list(read_json(processed_keyword_file))
    all_goods_info_dict = read_json(all_goods_file)
    all_keyword_list = []

    for user_name in user_name_list:
        all_records_file = f"{BASE_DIR}/{user_name}_record_info.json"
        record_data = read_json(all_records_file)
        keywords_list = find_key_values(record_data, 'keywords')
        keywords_list = [item for sublist in keywords_list for item in sublist] if keywords_list else []
        product_name_list = find_key_values(record_data, 'product_name')
        keyword_list = list(set(keywords_list + product_name_list))
        all_keyword_list.extend(keyword_list)
        print(f"用户 {user_name} 关键词列表长度 {len(keyword_list)}")

    useful_filed = [
        "itemName",
        'calTkRate',
        "calTkCommission",
        "finalPromotionPrice",
        "tkTotalSales",
        "biz365DayFuzzyString",
        "shopTitle",
        "whiteImage"
    ]

    all_keyword_list = list(set(all_keyword_list))
    print(
        f"所有用户关键词列表长度 {len(all_keyword_list)} 已经处理的关键词数量 {len(processed_keywords)} 已有商品信息数量 {len(all_goods_info_dict)}")

    save_counter = 0  # 新增：计数器
    for keyword in all_keyword_list:
        if keyword in processed_keywords:
            continue
        cookie_string = random.choice(cookie_string_list)
        goods = fetch_alimama_data(search_query=keyword, cookie_string=cookie_string)
        if goods is None:
            cookie_string_list.remove(cookie_string)
            print(f"关键词 '{keyword}' 抓取商品信息失败，移除 cookie: {cookie_string}")
            if not cookie_string_list:
                print("所有cookie均失效，停止抓取。")
                break
            continue

        print(f"关键词 '{keyword}' 抓取到 {len(goods)} 条商品信息。进度：{len(processed_keywords)}/{len(all_keyword_list)} \n")
        for good in goods:
            outputMktId = good.get('outputMktId', '')
            if outputMktId:
                good = {key: good[key] for key in useful_filed if key in good}
                incomeAmount = good.get('finalIncomeDTO', {}).get('incomeAmount', 0)
                commissionRate = good.get('finalIncomeDTO', {}).get('commissionRate', 0)

                if incomeAmount != 0:
                    good['calTkCommission'] = incomeAmount
                if commissionRate != 0:
                    good['calTkRate'] = commissionRate

                good['itemId'] = outputMktId
                good['updateTime'] = time.time()
            all_goods_info_dict[outputMktId] = good

        processed_keywords.append(keyword)
        save_counter += 1  # 每处理一个关键词就+1

        if save_counter >= 10:  # 每10个保存一次
            save_json_safe(all_goods_file, all_goods_info_dict)
            save_json_safe(processed_keyword_file, processed_keywords)
            save_counter = 0

    # 循环结束后，可能还有未保存的
    if save_counter > 0:
        save_json_safe(all_goods_file, all_goods_info_dict)
        save_json_safe(processed_keyword_file, processed_keywords)

def merge_all_goods(base_dir=BASE_DIR) -> str:
    """
    将 base_dir/taobao_goods 目录下的所有 CSV 文件合并为一个最终的 CSV 文件。
    只保留并重命名为英文的字段：
    ['商品id', '商品名称', '商品主图','店铺名称', '一级类目', '叶子类目', '活动到手价',
     '佣金率（%）', '佣金', '品牌', '优惠券面额',
     '淘宝客短链接(300天内有效)', '淘宝客链接', '淘口令(30天内有效)']
    对应英文列名为：
    ['product_id', 'product_name', 'main_image', 'shop_name', 'top_category',
     'leaf_category', 'promo_price', 'commission_rate_pct', 'commission', 'brand',
     'coupon_value', 'taobaoke_short_link_300d', 'taobaoke_link', 'taokouling_30d']
    返回输出文件的路径。
    """
    if not base_dir:
        raise ValueError("base_dir 不能为空")

    goods_file_dir = os.path.join(base_dir, "taobao_goods")
    if not os.path.isdir(goods_file_dir):
        raise FileNotFoundError(f"目录不存在: {goods_file_dir}")

    # 目标字段映射：中文 -> 英文
    mapping = {
        '商品id': 'outerId',
        '商品名称': 'goodsName',
        '商品主图': 'main_image',
        '店铺名称': 'shopName',
        '一级类目': 'top_category',
        '叶子类目': 'leaf_category',
        '活动到手价': 'promo_price',
        '佣金率（%）': 'commission_rate_pct',
        '佣金': 'commission',
        '品牌': 'brand',
        '优惠券面额': 'coupon_value',
        '淘宝客短链接(300天内有效)': 'taobaoke_short_link_300d',
        '淘宝客链接': 'taobaoke_link',
        '淘口令(30天内有效)': 'taokouling_30d',
    }

    # 获取所有 CSV 文件
    csv_files = glob.glob(os.path.join(goods_file_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"目录 {goods_file_dir} 中未找到 CSV 文件")

    dfs = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"读取文件失败，跳过 {fp}，错误: {e}")
            continue

        # 将中文列名映射为英文列名（如存在则重命名）
        rename_map = {cn: en for cn, en in mapping.items() if cn in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)

        # 确保最终输出列存在，缺失的填充为 NaN
        final_cols = list(mapping.values())
        for col in final_cols:
            if col not in df.columns:
                df[col] = pd.NA

        # 统一列顺序
        df = df[final_cols]
        dfs.append(df)

    if not dfs:
        raise ValueError("没有成功读取到任何数据")

    # 合并
    merged = pd.concat(dfs, ignore_index=True)
    merged['score'] = merged['commission_rate_pct'] * merged['commission'].fillna(0)
    # 生成一个新字段叫做all_str,由product_name， shop_name， top_category， leaf_category， brand拼接而成
    merged['all_str'] = merged['goodsName'].fillna('') + ' ' + merged['shopName'].fillna('') + ' ' + merged['top_category'].fillna('') + ' ' + merged['leaf_category'].fillna('') + ' ' + merged['brand'].fillna('')

    output_path = os.path.join(base_dir, "all_goods_info.csv")
    merged.to_csv(output_path, index=False)
    return output_path

def search_products(query, model, collection, top_n=5):
    """执行语义搜索"""
    # print(f"\n--- 正在搜索: '{query}' ---")
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n
    )

    formatted_results = []
    if results and results['ids'][0]:
        for i, item_id in enumerate(results['ids'][0]):
            distance = results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            formatted_results.append({
                "id": item_id,
                "metadata": metadata,
                "similarity": 1 - distance
            })

    return formatted_results

def search_goods(key_word_list=['零食']):
    """
    根据关键词搜索商品并返回商品列表。
    """
    result_list = []
    proxy = "http://127.0.0.1:7890"
    model_name = "BAAI/bge-base-zh-v1.5"
    db_path = "./product_db"
    collection_name = "my_products"
    base_dir = "goods_info"
    csv_path = f"{base_dir}/all_goods_info.csv"

    model, collection = init_model_and_db(
        model_name=model_name,
        db_path=db_path,
        collection_name=collection_name,
        device="cpu",
        proxy=proxy
    )

    add_products_from_csv(csv_path, model, collection)

    for q in key_word_list:
        search_results = search_products(q, model, collection, top_n=5)
        # print(f"{q} 搜索结果:\n{search_results}")
        result_list.extend(search_results if search_results else [])
    final_result_list = [result['metadata'] for result in result_list if 'metadata' in result]
    # 按照outerId进行去重
    unique_outer_ids = set()
    final_result_list = [item for item in final_result_list if item['outerId'] not in unique_outer_ids and not unique_outer_ids.add(item['outerId'])]

    return final_result_list


def add_image_info():
    """
    下载商品图片，如果成功，则将图片的绝对路径更新到CSV文件中。
    """
    base_dir = "goods_info"
    csv_path = f"{base_dir}/all_goods_info.csv"
    images_dir = os.path.join(base_dir, "images")

    # 1. 确保图片存储目录存在
    os.makedirs(images_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: CSV文件未找到于 '{csv_path}'")
        return

    # 2. 检查并添加 abd_image_path 列（如果不存在）
    if 'abd_image_path' not in df.columns:
        df['abd_image_path'] = pd.NA

    # 3. 遍历每一行，下载图片并更新路径
    print("开始处理图片下载和路径更新...")
    for index, row in df.iterrows():
        image_url = row.get('main_image')
        outer_id = row.get('outerId', f'行_{index}')  # 获取outerId用于命名，如果不存在则用行号

        if pd.isna(image_url) or not image_url:
            continue

        try:
            # 修正协议头
            if not image_url.startswith(('http:', 'https:')):
                image_url = f'https:{image_url}'

            image_name = f"{outer_id}.jpg"
            # 这里创建的是相对路径
            relative_image_path = os.path.join(images_dir, image_name)
            if os.path.exists(relative_image_path):
                print(f"跳过: {image_name} 已存在。")
                continue

            # 下载图片 (将路径字符串转换为Path对象)
            response = download_public_image(image_url, pathlib.Path(relative_image_path))

            # 4. 如果下载成功，更新DataFrame中的绝对路径
            if response is True:
                # 获取文件的绝对路径
                absolute_path = os.path.abspath(relative_image_path)
                # 使用 .at 高效地为单元格赋值
                df.at[index, 'abd_image_path'] = absolute_path
                print(f"成功: {image_name} 的路径已更新。")
            else:
                print(f"失败: {image_name}, URL: {image_url}, 原因: {response}")

        except Exception as e:
            print(f"处理商品 {outer_id} 时发生未知错误: {e}")

    # 5. 所有行处理完毕后，将更新后的DataFrame保存回原文件
    try:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n处理完成！已将更新后的数据保存回: {csv_path}")
    except Exception as e:
        print(f"保存CSV文件时发生错误: {e}")


if __name__ == "__main__":
    # add_image_info()

    # merge_all_goods()


    result_list = search_goods([
                        "电竞零食",
                        "开黑必备",
                        "游戏夜宵",
                        "懒人速食"
                    ])
    print(result_list)
    # print(goods_infos)

