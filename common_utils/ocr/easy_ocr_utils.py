import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Tuple, Dict, Any, Optional


# --- 1. 模型初始化模块 (无变化) ---
def init_ocr_model() -> PaddleOCR:
    """
    初始化并返回 PaddleOCR 模型实例。
    """
    print("正在初始化 PaddleOCR 模型...")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='ch'
    )
    print("PaddleOCR 模型初始化完成。")
    return ocr


# --- 2. 图像处理模块 (无变化) ---
def load_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int]]:
    """
    从指定路径加载图片，并返回图片对象、高度和宽度。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法从路径加载图片: {image_path}")
        return None, None, None
    height, width, _ = image.shape
    print(f"图片加载成功: {image_path}, 尺寸: {width}x{height}")
    return image, height, width


# --- 3. OCR预测模块 (无变化) ---
def predict_text(ocr_model: PaddleOCR, image: np.ndarray) -> List[Dict[str, Any]]:
    """
    使用OCR模型对图像进行文本预测。
    """
    print("正在进行OCR识别...")
    result = ocr_model.predict(input=image)
    print("OCR识别完成。")
    return result


# --- 4. 字幕定位模块 (合并后的新函数) ---
def find_subtitle(ocr_result: List[Dict[str, Any]], image_height: int, image_width: int) -> Optional[np.ndarray]:
    """
    从OCR结果中定位字幕框。
    此函数内部完成候选框筛选、评分和选择的全过程。

    :param ocr_result: PaddleOCR的原始预测结果。
    :param image_height: 图片高度。
    :param image_width: 图片宽度。
    :return: 最终确定的字幕框坐标（numpy数组），或None。
    """
    # 步骤1: 筛选位于画面底部的候选框
    if not ocr_result or 'rec_polys' not in ocr_result[0] or ocr_result[0]['rec_polys'] is None:
        return None

    all_boxes = ocr_result[0]['rec_polys']
    bottom_threshold_y = image_height * 0.7
    candidate_boxes = [box for box in all_boxes if np.min(box[:, 1]) > bottom_threshold_y]

    # 如果没有候选框，直接返回
    if not candidate_boxes:
        return None

    # 如果只有一个候选框，直接返回
    if len(candidate_boxes) == 1:
        print("找到唯一候选字幕框。")
        return candidate_boxes[0]

    # 步骤2: 对多个候选框进行评分和选择
    print(f"找到 {len(candidate_boxes)} 个候选框，正在进行评分筛选...")

    # 定义评分权重
    Y_POSITION_WEIGHT = 1.0
    X_CENTERING_WEIGHT = 0.5
    image_center_x = image_width / 2

    def calculate_score(box: np.ndarray) -> float:
        """计算单个框的得分"""
        center_y = np.mean(box[:, 1])
        y_score = (center_y / image_height) * Y_POSITION_WEIGHT

        center_x = np.mean(box[:, 0])
        distance_from_center = abs(center_x - image_center_x)
        x_penalty = (distance_from_center / image_width) * X_CENTERING_WEIGHT

        return y_score - x_penalty

    # 使用max函数和评分函数直接找到最佳的box
    best_box = max(candidate_boxes, key=calculate_score)

    return best_box


# --- 5. 可视化模块 (无变化) ---
def draw_box_on_image(image: np.ndarray, box: Optional[np.ndarray], output_path: str):
    """
    在图片上绘制给定的框，并保存到文件。
    """
    if box is None:
        print("没有定位到字幕框，不进行绘制。")
        return

    points = box.astype(np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(output_path, image)
    print(f"结果已绘制并保存到: {output_path}")


# --- 6. 主流程控制 (调用方式简化) ---
def main(image_path: str, output_path: str):
    """
    主函数，执行完整的字幕定位和可视化流程。

    :param image_path: 输入图片的路径。
    :param output_path: 输出图片的保存路径。
    """
    # 步骤1: 初始化OCR模型
    ocr_model = init_ocr_model()

    # 步骤2: 加载图片
    image, height, width = load_image(image_path)
    if image is None:
        return

    # 步骤3: 执行OCR预测
    ocr_result = predict_text(ocr_model, image)

    # --- 关键变化：一步到位寻找字幕 ---
    # 步骤4: 从OCR结果中寻找字幕
    subtitle_box = find_subtitle(ocr_result, height, width)

    # 步骤5: 打印结果并进行可视化
    print("\n" + "=" * 20 + " 定位结果 " + "=" * 20)
    if subtitle_box is not None:
        print("✅ 成功定位到字幕！")
        print("字幕坐标为:")
        print(subtitle_box.tolist())  # 打印时转为list
        draw_box_on_image(image.copy(), subtitle_box, output_path)
    else:
        print("❌ 未能在图片中定位到字幕。")
    print("=" * 54)


# --- 程序入口 ---
if __name__ == "__main__":
    INPUT_IMAGE_PATH = "frame_0001.jpg"
    OUTPUT_IMAGE_PATH = "frame_0001_result.jpg"

    try:
        main(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)
    except Exception as e:
        print(f"程序运行期间发生未处理的异常: {e}")