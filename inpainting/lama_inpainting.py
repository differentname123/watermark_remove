import time
import os
from collections import Counter, defaultdict

from simple_lama_inpainting import SimpleLama
from PIL import Image



if __name__ == "__main__":
    lama = SimpleLama()  # 或者 SimpleLama(device=torch.device("cuda")) 如果有GPU
    images = [Image.open('a.jpg')]
    masks = [Image.open('mask_image.jpg')]

    # 复制images便于测试
    images = images * 10
    masks = masks * 10

    start_time = time.time()
    for i in range(len(images)):
        image = images[i]
        mask = masks[i]
        result = lama.inpaint(image, mask)
        result.save(f"output/result_{i + 2}.jpg")
    print("Batch processing time:", time.time() - start_time)



    start_time = time.time()
    result_images = lama.inpaint_batch(images, masks)
    for idx, res in enumerate(result_images):
        res.save(f"output/result_{idx + 2}_batch.jpg")
    print("Batch processing time:", time.time() - start_time)
