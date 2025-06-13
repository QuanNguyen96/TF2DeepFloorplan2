import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_black_pixels(image_path, threshold=30, min_region_size=10):
    """
    Trích xuất pixel gần màu đen và loại bỏ các cụm nhỏ hơn ngưỡng.
    
    Args:
        image_path: Đường dẫn ảnh đầu vào.
        threshold: Ngưỡng độ đen (0-255).
        min_region_size: Số pixel nhỏ nhất để giữ lại 1 vùng liên thông.
        
    Returns:
        image: ảnh gốc
        clean_mask: mask nhị phân (0 hoặc 1) sau khi đã loại nhiễu nhỏ.
    """
    # Đọc ảnh RGB
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không đọc được ảnh!")

    # Tính khoảng cách màu với đen
    distance = np.linalg.norm(image.astype(np.int16), axis=2)
    mask = (distance < threshold).astype(np.uint8)

    # Gán nhãn các vùng liên thông
    num_labels, labels = cv2.connectedComponents(mask)

    # Tạo mask mới và giữ lại vùng đủ lớn
    clean_mask = np.zeros_like(mask)
    for label in range(1, num_labels):  # Bỏ label 0 (nền)
        region_size = np.sum(labels == label)
        if region_size >= min_region_size:
            clean_mask[labels == label] = 1

    return image, clean_mask

# Đường dẫn ảnh
# image_path = "./resources/30939153.jpg"
image_path = "./resources/123.jpg"

# Trích xuất và lọc nhiễu
image, clean_mask = extract_black_pixels(image_path, threshold=30, min_region_size=30)

# Hiển thị kết quả
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Ảnh gốc")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Pixel đen sau khi lọc")
plt.imshow(clean_mask, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
