import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_windows

def detect_wall_segments(image_path, patch_size=32, var_thresh=500, min_length=10):
    # Step 1: Load grayscale image and binarize (ngưỡng đảo: tường thành pixel trắng)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Cắt từng patch nhỏ và tính variance
    pad_h = patch_size - binary.shape[0] % patch_size
    pad_w = patch_size - binary.shape[1] % patch_size
    padded = np.pad(binary, ((0, pad_h), (0, pad_w)), mode='constant')
    windows = view_as_windows(padded, (patch_size, patch_size), step=patch_size)
    score_mask = np.zeros_like(padded)

    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            patch = windows[i, j]
            if np.var(patch) > var_thresh:
                score_mask[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = 255

    score_mask = score_mask[:binary.shape[0], :binary.shape[1]]

    # Step 3: Morphological filtering để nối liền tường
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(score_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 4: Hough Transform tìm đoạn thẳng
    lines = cv2.HoughLinesP(morph, 1, np.pi / 180, threshold=80, minLineLength=min_length, maxLineGap=5)

    # Step 5: Vẽ các đoạn line
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    line_segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
            line_segments.append(((x1, y1), (x2, y2)))

    return vis, line_segments

# Dùng thử
img_path = "./resources/123.jpg"
vis, segments = detect_wall_segments(img_path)
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.title(f"Detected {len(segments)} wall segments")
plt.axis("off")
plt.show()
