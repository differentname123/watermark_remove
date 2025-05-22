import easyocr
import cv2
import numpy as np

# 1. 初始化 OCR
reader = easyocr.Reader(['ch_sim', 'en'])  # 中文简体 + 英文

# 2. 运行 OCR
image_path = 'a.jpg'
results = reader.readtext(image_path, detail=1)

# 3. 读取原始图像
image = cv2.imread(image_path)

# 4. 在图像上绘制识别结果
for bbox, text, confidence in results:
    # bbox 是 4 个顶点 (top-left, top-right, bottom-right, bottom-left)
    pts = np.array(bbox, dtype=np.int32).reshape((-1,1,2))
    # 4.1 画多边形框
    cv2.polylines(image, [pts], isClosed=True, color=(0,255,0), thickness=2)
    # 4.2 在框左上角上方写上文字
    tl = bbox[0]
    org = (int(tl[0]), int(tl[1]) - 10)  # 文本起点
    cv2.putText(
        image, text, org,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0,0,255),
        thickness=1,
        lineType=cv2.LINE_AA
    )

# 5. 保存结果图
output_path = 'result.jpg'
cv2.imwrite(output_path, image)
print(f"已生成带识别框的新图：{output_path}")