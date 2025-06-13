import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def smooth_edges_with_hough(image, canny_thresh1=50, canny_thresh2=150, hough_thresh=50):
    # Chuyển ảnh về uint8 nếu cần (mpimg trả về float32 từ 0-1)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Nếu là ảnh màu (RGB), chuyển về grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Bước 1: Tìm biên bằng Canny
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)

    # Bước 2: Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)

    # Bước 3: Vẽ các đường lên ảnh trắng
    hough_mask = np.zeros_like(gray)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 1)

    return (hough_mask > 0).astype(np.uint8)

# Đường dẫn ảnh
img_path = "./resources/30939153.jpg"
image = mpimg.imread(img_path)

# Làm mịn bằng Hough + Canny
smooth_mask = smooth_edges_with_hough(image)

# Hiển thị kết quả
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Ảnh gốc")

plt.subplot(1, 2, 2)
plt.imshow(smooth_mask, cmap='gray')
plt.title("Mask Hough + Canny")

plt.tight_layout()
plt.show()
