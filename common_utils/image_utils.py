import cv2
import numpy as np

def denormalize_bbox(bbox_norm, image_width, image_height, to_int=True):
    """
    将归一化的边界框坐标转换为实际图像坐标。

    参数：
        bbox_norm: list or tuple, 归一化坐标 [x_min, y_min, x_max, y_max]，取值范围在 0~1 之间
        image_width: int, 图像的实际宽度
        image_height: int, 图像的实际高度
        to_int: bool, 是否将输出坐标转换为整数，默认 True

    返回：
        list: 实际图像坐标 [x_min, y_min, x_max, y_max]
    """
    x_min = bbox_norm[0] * image_width
    y_min = bbox_norm[1] * image_height
    x_max = bbox_norm[2] * image_width
    y_max = bbox_norm[3] * image_height

    if to_int:
        return [int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))]
    else:
        return [x_min, y_min, x_max, y_max]


def select_region_and_create_mask(image_path, window_width=800, window_height=600):
    """
    加载图片，若图片尺寸过大，则生成一个缩略图用于交互显示，而不改变原图；
    用户在缩略图上选择两个点（左上角和右下角），程序将缩略图中的坐标
    映射回原图坐标，从而生成与原图尺寸一致的 mask 图片（选中区域像素为 255，其余为 0）。

    参数:
        image_path: 图片的文件路径。
        window_width: 交互窗口期望的宽度（单位像素）。
        window_height: 交互窗口期望的高度（单位像素）。

    返回:
        mask: 与原图尺寸一致的 mask 图像（灰度图），若选择操作未完成则返回 None。
    """
    # 加载原始图片
    img = cv2.imread(image_path)
    if img is None:
        print("错误：无法加载图片！")
        return None

    orig_h, orig_w = img.shape[:2]
    print(f"原图尺寸: {orig_w}x{orig_h}")

    # 计算缩放因子，确保整张图片都能显示在指定窗口范围内（不放大）
    scale = min(window_width / orig_w, window_height / orig_h, 1.0)
    if scale < 1.0:
        # 使用最近邻插值，避免缩小后出现模糊现象
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
        display_img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        print(f"生成了缩略图，缩放因子: {scale:.2f}，显示尺寸: {disp_w}x{disp_h}")
    else:
        scale = 1.0  # 图片尺寸本身就在窗口尺寸内
        display_img = img.copy()
        print("图片尺寸未超过窗口设定，无需缩放。")

    clone = display_img.copy()
    points = []

    def click_event(event, x, y, flags, param):
        nonlocal points, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # 在点击处绘制一个小圆点提示用户当前的点击位置
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", clone)
            # 如果已点击两次，则画出矩形
            if len(points) == 2:
                pt1 = points[0]
                pt2 = points[1]
                x1_disp = min(pt1[0], pt2[0])
                y1_disp = min(pt1[1], pt2[1])
                x2_disp = max(pt1[0], pt2[0])
                y2_disp = max(pt1[1], pt2[1])
                cv2.rectangle(clone, (x1_disp, y1_disp), (x2_disp, y2_disp), (0, 0, 255), 2)
                cv2.imshow("Image", clone)

    # 创建窗口（WINDOW_AUTOSIZE 确保窗口大小按照图片设置，且不允许手动拖动改变）
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Image", clone)
    cv2.setMouseCallback("Image", click_event)

    print("请依次点击图片中的左上角和右下角位置，或按 'q' / Esc 退出。")

    # 等待用户操作
    while True:
        key = cv2.waitKey(1) & 0xFF
        # 按 q 或 Esc 退出程序
        if key == ord('q') or key == 27:
            break
        # 如果已选择两个点，则退出
        if len(points) >= 2:
            break

    cv2.destroyWindow("Image")

    if len(points) < 2:
        print("错误：未正确选择两个点。")
        return None

    # 将缩略图的坐标映射回原图坐标
    pt1_disp, pt2_disp = points[0], points[1]
    x1_disp, y1_disp = pt1_disp
    x2_disp, y2_disp = pt2_disp
    x1 = int(min(x1_disp, x2_disp) / scale)
    y1 = int(min(y1_disp, y2_disp) / scale)
    x2 = int(max(x1_disp, x2_disp) / scale)
    y2 = int(max(y1_disp, y2_disp) / scale)

    bbox_norm = [0.800000, 0.168000, 0.871000, 0.225000]
    width = 1037
    height = 1247

    bbox_pixel = denormalize_bbox(bbox_norm, width, height)

    x1,y1,x2,y2 = bbox_pixel



    print(f"映射回原图的ROI坐标: ({x1}, {y1}) 到 ({x2}, {y2})")

    # 生成与原图尺寸一致的 mask，选择区域像素设置为 255，其余为 0
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    return mask


if __name__ == "__main__":
    image_path = "../inpainting/a.jpg"  # 请替换为你的图片文件路径
    mask_image = select_region_and_create_mask(image_path, window_width=800, window_height=600)

    if mask_image is not None:
        # 显示并保存mask图
        cv2.imshow("Mask", mask_image)
        cv2.imwrite("../inpainting/mask_image.jpg", mask_image)
        print("mask_image.jpg 已保存。")

        # 生成原图上掩码位置涂白后的图片
        original_img = cv2.imread(image_path)
        if original_img is None:
            print("错误：无法加载原始图片！")
        else:
            white_masked_image = original_img.copy()
            # 将掩码区域的像素全部设置为白色（BGR格式下白色为 [255, 255, 255]）
            white_masked_image[mask_image == 255] = [255, 255, 255]
            cv2.imshow("White Masked Image", white_masked_image)
            cv2.imwrite("../inpainting/white_masked_image.jpg", white_masked_image)
            print("white_masked_image.jpg 已保存。")

        print("按任意键退出。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()