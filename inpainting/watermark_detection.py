import torch
from PIL import Image
from ultralytics import YOLO

# -------------------------------------------------
# 1. 设备与模型初始化
# -------------------------------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# 加载权重
yolo_model = YOLO("yolo11x-train28-best.pt")

# 将模型迁移到 GPU，并做推理加速配置
if DEVICE.startswith("cuda"):
    yolo_model.to(DEVICE)      # 把权重放入显存
    yolo_model.fuse()          # 融合 Conv + BN，减少显存访问
    yolo_model.half()          # FP16 推理（节省显存、加速）

# -------------------------------------------------
# 2. 推理函数
# -------------------------------------------------
@torch.inference_mode()  # 等价于 `with torch.no_grad()`
def yolo_predict(image: Image.Image) -> Image.Image:
    """
    利用 YOLO 模型对图像进行目标检测，
    在控制台打印每个检测框的位置及信息，
    并返回绘制了检测框的图像。
    """
    # 推理
    results = yolo_model(
        image,
        imgsz=1024,
        augment=True,
        iou=0.5
    )

    assert len(results) == 1, "Expected one result from YOLO detection."
    result = results[0]

    # -----------------------------------------------------------------
    # 打印每个检测框的位置、类别、置信度
    # -----------------------------------------------------------------
    # boxes.xyxy 是一个 Tensor of shape (n,4)
    # boxes.conf 是 (n,), boxes.cls 是 (n,)
    xyxys  = result.boxes.xyxy.cpu().tolist()
    confs  = result.boxes.conf.cpu().tolist()
    classes= result.boxes.cls.cpu().tolist()
    for i, (xyxy, conf, cls) in enumerate(zip(xyxys, confs, classes)):
        x1, y1, x2, y2 = xyxy
        cls = int(cls)
        class_name = yolo_model.names.get(cls, str(cls))
        print(f"[INFO] Box {i}: class={class_name} (id={cls}), "
              f"conf={conf:.2f}, "
              f"xyxy=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    # 获取绘制了检测框的图像 (BGR)
    im_array = result.plot()
    # 转为 PIL RGB
    im_rgb = Image.fromarray(im_array[..., ::-1])
    return im_rgb

# -------------------------------------------------
# 3. 主程序：读取图片并另存含检测框的信息图像
# -------------------------------------------------
if __name__ == "__main__":
    # 修改为你的输入图片路径
    input_image_path = "../inpainting/input.jpg"
    # 修改为你希望保存输出图片的路径
    output_image_path = "../inpainting/output.jpg"

    # 打开输入图片
    image_input = Image.open(input_image_path)

    # 调用预测函数
    result_image = yolo_predict(image_input)

    # 保存带有检测框的图片
    result_image.save(output_image_path)
    print(f"[INFO] Saved output image at: {output_image_path}")