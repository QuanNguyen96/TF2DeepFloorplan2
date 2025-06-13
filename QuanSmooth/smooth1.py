import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# def find_rectangles_global_greedy(mask, c=1, ratio_thresh=1.0, min_area_required=4):
#     h, w = mask.shape
#     used = np.zeros_like(mask, dtype=bool)
#     candidate_rects = []

#     def is_valid_block(x1, y1, x2, y2):
#         if x2 >= w or y2 >= h:
#             return False
#         for yy in range(y1, y2 + 1, c):
#             for xx in range(x1, x2 + 1, c):
#                 if mask[yy, xx] != 1:
#                     return False
#         return True

#     # Bước 1: Tìm tất cả các hình chữ nhật hợp lệ
#     for y in range(h):
#         for x in range(w):
#             if mask[y, x] != 1:
#                 continue

#             # === Mở rộng theo X trước ===
#             max_x = x
#             while max_x + c < w and mask[y, max_x + c] == 1:
#                 max_x += c
#             for x2 in range(x, max_x + 1, c):
#                 max_y = y
#                 while max_y + c < h and is_valid_block(x, y, x2, max_y + c):
#                     max_y += c
#                 width = x2 - x + 1
#                 height = max_y - y + 1
#                 area = width * height
#                 if area >= min_area_required:
#                     ratio = max(width / height, height / width)
#                     if ratio >= ratio_thresh:
#                         length = max(width, height)
#                         score = length * 10000 + area
#                         candidate_rects.append((score, (x, y, x2, max_y)))

#             # === Mở rộng theo Y trước ===
#             max_y = y
#             while max_y + c < h and mask[max_y + c, x] == 1:
#                 max_y += c
#             for y2 in range(y, max_y + 1, c):
#                 max_x = x
#                 while max_x + c < w and is_valid_block(x, y, max_x + c, y2):
#                     max_x += c
#                 width = max_x - x + 1
#                 height = y2 - y + 1
#                 area = width * height
#                 if area >= min_area_required:
#                     ratio = max(width / height, height / width)
#                     if ratio >= ratio_thresh:
#                         length = max(width, height)
#                         score = length * 10000 + area
#                         candidate_rects.append((score, (x, y, max_x, y2)))

#     # Bước 2: Sắp xếp các hình chữ nhật theo độ ưu tiên (dài nhất trước)
#     candidate_rects.sort(reverse=True)

#     selected_rects = []
#     for _, (x1, y1, x2, y2) in candidate_rects:
#         valid = True
#         for yy in range(y1, y2 + 1, c):
#             for xx in range(x1, x2 + 1, c):
#                 if used[yy, xx]:
#                     valid = False
#                     break
#             if not valid:
#                 break
#         if valid:
#             selected_rects.append((x1, y1, x2, y2))
#             for yy in range(y1, y2 + 1, c):
#                 for xx in range(x1, x2 + 1, c):
#                     used[yy, xx] = True

#     # Tạo filtered_mask
#     filtered_mask = np.zeros_like(mask, dtype=int)
#     for x1, y1, x2, y2 in selected_rects:
#         for yy in range(y1, y2 + 1, c):
#             for xx in range(x1, x2 + 1, c):
#                 filtered_mask[yy, xx] = 1

#     return filtered_mask, selected_rects

# def show_result(mask, filtered_mask, rectangles):
#     plt.figure(figsize=(15, 5))

#     # 1. Ảnh gốc
#     plt.subplot(1, 3, 1)
#     plt.imshow(mask, cmap='gray')
#     plt.title("Ảnh gốc")
#     plt.axis('off')

#     # 2. Ảnh sau khi lọc
#     plt.subplot(1, 3, 2)
#     plt.imshow(filtered_mask, cmap='gray')
#     plt.title("Ảnh lọc hình chữ nhật")
#     plt.axis('off')

#     # 3. Ảnh + viền đỏ pixel đúng
#     plt.subplot(1, 3, 3)
#     plt.imshow(mask, cmap='gray')
#     ax = plt.gca()
#     for x1, y1, x2, y2 in rectangles:
#         rect = patches.Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1 + 1, y2 - y1 + 1,
#                                  linewidth=2, edgecolor='red', facecolor='none')
#         ax.add_patch(rect)
#     plt.title("Ảnh + viền đỏ hình chữ nhật")
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

# # =====================
# # === MAIN PROGRAM ====
# # =====================
# import matplotlib.patches as patches
# # ✅ 1. Load từ CSV (thay bằng tên file thật của bạn)
# mask = np.loadtxt("wall1.csv",skiprows=1, delimiter=",", dtype=int)
# mask = np.array([
#     [0,0,0,0,0, 0,0,0,0,0,  0,0,0,0,0, 0,0,0,0,0],
#     [0,1,1,1,1, 0,0,0,0,0,  0,0,1,1,1, 1,0,0,0,0],
#     [0,1,1,1,1, 0,0,0,0,0,  0,0,1,1,0, 1,0,0,0,0],
#     [0,1,1,1,1, 0,0,1,1,1,  1,1,1,1,1, 1,0,0,0,0],
#     [0,1,1,1,1, 0,0,1,1,1,  1,1,1,1,1, 1,0,0,0,0],
#     [0,0,0,0,0, 0,0,1,1,1,  1,1,0,0,0, 0,0,0,0,0], 
#     [0,0,0,0,0, 0,0,1,1,1,  1,1,0,0,0, 0,0,0,0,0],
#     [0,0,0,0,0, 0,0,1,1,1,  1,1,0,0,0, 0,0,0,0,0],
#     [0,0,0,0,0, 0,0,0,0,0,  0,0,0,0,0, 0,0,0,0,0],
#     [0,0,0,0,0, 0,0,0,0,0,  0,0,0,0,0, 0,0,0,0,0],
# ])

# filtered_mask, rects = find_rectangles_global_greedy(mask, c=1, ratio_thresh=1.0, min_area_required=4)

# show_result(mask, filtered_mask, rects)


import numpy as np
import cv2
from shapely.geometry import LineString
import matplotlib.pyplot as plt

# Load mask
mask = np.loadtxt("wall1.csv", skiprows=1, delimiter=",", dtype=int).astype(np.uint8)
original_mask = mask.copy()

# Find contours
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# RGB image to draw outline
outline_img = np.stack([mask * 255] * 3, axis=-1)

# New mask to fill simplified regions
filled_mask = np.zeros_like(mask)

for c in cnts:
    if cv2.arcLength(c, True) < 40:
        continue
    line = LineString([pt[0] for pt in c]).simplify(2)
    coords = np.array(list(line.coords), dtype=np.int32).reshape((-1, 1, 2))

    # Vẽ contour đơn giản hóa
    cv2.polylines(outline_img, [coords], isClosed=True, color=(0, 255, 0), thickness=1)

    # Fill vùng bên trong contour đã đơn giản
    cv2.fillPoly(filled_mask, [coords], 1)

# Hiển thị
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Mask")
plt.imshow(original_mask, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Simplified Contours")
plt.imshow(outline_img)

plt.subplot(1, 3, 3)
plt.title("Filled Simplified Mask")
plt.imshow(filled_mask, cmap='gray')

plt.tight_layout()
plt.show()
