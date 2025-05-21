import time
import os
from collections import Counter, defaultdict

from simple_lama_inpainting import SimpleLama
from PIL import Image

def load_image_and_mask():
    """
    加载图像和对应的掩码
    """
    image_path = "LaMa_test_images"
    image_and_mask_map = {}
    # 遍历image_path下所有的png文件
    png_files = [f for f in os.listdir(image_path) if f.endswith('.png') and 'mask' not in f and 'inpainted' not in f]
    mask_png_files = [f for f in os.listdir(image_path) if f.endswith('.png') and 'mask' in f and 'inpainted' not in f]
    for png_file in png_files:
        mask_png_file = png_file.replace('.png', '_mask.png')
        if mask_png_file in mask_png_files:
            image_and_mask_map[png_file] = mask_png_file
    print("待处理图片数量:", len(image_and_mask_map))


    start_time = time.time()

    # 第一步：扫描所有文件，按尺寸收集匹配对
    size_counter = Counter()
    size_to_pairs = defaultdict(list)

    for image_file, mask_file in image_and_mask_map.items():
        img_full = os.path.join(image_path, image_file)
        msk_full = os.path.join(image_path, mask_file)

        # 打开但不加载全部像素，只取尺寸
        with Image.open(img_full) as img:
            img_size = img.size  # (width, height)
        with Image.open(msk_full) as msk:
            msk_size = msk.size

        # 只保留尺寸完全相同的对
        if img_size == msk_size:
            size_counter[img_size] += 1
            size_to_pairs[img_size].append((image_file, mask_file))

    # 如果没有任何尺寸匹配的对，则退出
    if not size_counter:
        raise RuntimeError("没有找到尺寸匹配的图像/掩码对。")

    # 找到出现次数最多的尺寸
    best_size, best_count = size_counter.most_common(1)[0]
    print(f"最佳尺寸: {best_size}，共 {best_count} 对，载入这一组，丢弃其它。")

    # 第二步：只加载最佳尺寸的图像和掩码
    images = []
    masks = []

    for image_file, mask_file in size_to_pairs[best_size]:
        img_full = os.path.join(image_path, image_file)
        msk_full = os.path.join(image_path, mask_file)

        # 这里我们真正加载并复制到内存
        image = Image.open(img_full).convert("RGB").copy()
        mask = Image.open(msk_full).convert("L").copy()

        images.append(image)
        masks.append(mask)

    end_time = time.time()
    print(f"总共加载 {len(images)} 张图像和掩码，用时 {end_time - start_time:.2f} 秒")
    return images, masks

if __name__ == "__main__":
    lama = SimpleLama()  # 或者 SimpleLama(device=torch.device("cuda")) 如果有GPU
    # images, masks = load_image_and_mask()

    images = [Image.open('a1.jpg')]
    masks = [Image.open('mask_image.jpg')]

    start_time = time.time()
    for i in range(len(images)):
        image = images[i]
        mask = masks[i]
        result = lama.inpaint(image, mask)
        result.save(f"output/result_{i + 1}.jpg")
    print("Batch processing time:", time.time() - start_time)

    start_time = time.time()
    result_images = lama.inpaint_batch(images, masks)
    for idx, res in enumerate(result_images):
        res.save(f"output/result_{idx + 1}_batch.jpg")
    print("Batch processing time:", time.time() - start_time)
