import numpy as np
import matplotlib.pyplot as plt
import tempfile
from dfp.net import *
from dfp.data import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import groupby
from argparse import Namespace
import io
import os
import gc
import cv2
import json


def preprocess_wall_map(wall_map):
    """Làm mịn ảnh tường bằng phép closing (lấp lỗ)"""
    img = (wall_map * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed

def extract_large_contours(binary_img, min_area=50):
    """Lấy contour có diện tích lớn, bỏ phần lồi nhỏ như mép cửa"""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    return large_contours

def make_contour_axis_aligned(contour, epsilon=3):
    """Xấp xỉ contour vuông góc và căn chỉnh theo trục X hoặc Z"""
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    aligned = []

    for pt in approx:
        aligned.append(pt[0].copy())

    for i in range(len(aligned) - 1):
        p1 = aligned[i]
        p2 = aligned[i + 1]

        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        if dx > dy:
            p2[1] = p1[1]  # Ép ngang
        elif dy > dx:
            p2[0] = p1[0]  # Ép dọc
        # Nếu gần bằng nhau thì giữ nguyên (để tránh phá đường nghiêng nhẹ)

    return np.array(aligned, dtype=np.int32).reshape(-1, 1, 2)


def draw_clean_walls(shape, contours, epsilon=3, min_area=50):
    """Vẽ lại các tường đã làm sạch và vuông hóa"""
    canvas = np.zeros(shape, dtype=np.uint8)
    epsilon_2=epsilon
    for cnt in contours:
       # Tạo canvas hiển thị debug cho từng contour
        debug_canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        cv2.drawContours(debug_canvas, [cnt], -1, (0, 255, 0), 1)  # Vẽ contour màu xanh

        plt.figure(figsize=(10, 10))
        plt.imshow(debug_canvas)
        plt.title("Contour trước khi vuông hóa")
        plt.show()
        print("area=",cv2.contourArea(cnt))
        if cv2.contourArea(cnt) < min_area:
            continue
        if cv2.contourArea(cnt) > 1000:
          epsilon_2 = 2.4
        aligned_cnt = make_contour_axis_aligned(cnt, epsilon_2)
        cv2.drawContours(canvas, [aligned_cnt], -1, 255, thickness=-1)
    return (canvas > 0).astype(np.uint8)

def straighten_wall_map(wall_map, epsilon=3, min_area=50):
    """Pipeline xử lý toàn diện: mịn hóa + lọc nhiễu + vuông hóa"""
    processed = preprocess_wall_map(wall_map)
    contours = extract_large_contours(processed, min_area=min_area)
    straightened = draw_clean_walls(processed.shape, contours, epsilon=epsilon, min_area=min_area)
    
    return straightened


def smooth_map_cv2(grid, method='open', structure_size=3):
    kernel = np.ones((structure_size, structure_size), np.uint8)

    if method == 'open':
        result = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    elif method == 'close':
        result = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    elif method == 'both':
        opened = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("method phải là 'open', 'close' hoặc 'both'")

    return result.astype(int)


def find_horizontal_rectangles(mask, min_len=2):
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    rectangles = []

    for i in range(H):
        j = 0
        while j < W:
            # Nếu là điểm bắt đầu 1 chuỗi liên tiếp chưa vẽ
            if mask[i, j] == 1 and not visited[i, j]:
                start = j
                while j < W and mask[i, j] == 1 and not visited[i, j]:
                    j += 1
                end = j
                length = end - start
                if length >= min_len:
                    rectangles.append((i, start, 1, length))  # (y, x, h=1, w)
                    visited[i, start:end] = True
            else:
                j += 1
    return rectangles

def unique_points_with_min_max(points):
    seen = set()
    unique_points = []
    min_i, max_i = float('inf'), float('-inf')
    min_j, max_j = float('inf'), float('-inf')

    for i, j in points:
        if (i,j) not in seen:
            seen.add((i,j))
            unique_points.append((i,j))
            if i < min_i: min_i = i
            if i > max_i: max_i = i
            if j < min_j: min_j = j
            if j > max_j: max_j = j

    return unique_points, min_i, max_i, min_j, max_j
def merge_consecutive(sorted_coords):
    ranges = []
    start = sorted_coords[0]
    prev = start
    for x in sorted_coords[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = x
            prev = x
    ranges.append((start, prev))
    return ranges

def find_consecutive_ranges(arr):
    """
    arr: numpy 2D binary array (0/1)
    Return:
        - List các dict {start: [i,j], end: [i,j]} thể hiện vùng liên tiếp theo hàng hoặc cột
        - min_i, max_i, min_j, max_j các giá trị min max
    """
    points = list(zip(*np.where(arr == 1)))
    if not points:
        return [], 0, 0, 0, 0

    unique_points, min_i, max_i, min_j, max_j = unique_points_with_min_max(points)

    # Nhóm theo hàng (i)
    group_by_i = {}
    for i, j in unique_points:
        group_by_i.setdefault(i, []).append(j)

    # Nhóm theo cột (j)
    group_by_j = {}
    for i, j in unique_points:
        group_by_j.setdefault(j, []).append(i)

    result_set = set()

    # Gom dải j liên tiếp theo i
    for i, js in group_by_i.items():
        sorted_js = sorted(js)
        j_ranges = merge_consecutive(sorted_js)
        for j_start, j_end in j_ranges:
            result_set.add(( (i,j_start), (i,j_end) ))

    # Gom dải i liên tiếp theo j
    for j, is_ in group_by_j.items():
        sorted_is = sorted(is_)
        i_ranges = merge_consecutive(sorted_is)
        for i_start, i_end in i_ranges:
            if i_start != i_end:  # tránh vùng 1 điểm trùng lặp với trên
                key = ((i_start,j), (i_end,j))
                if key not in result_set:
                    result_set.add(key)

    # Chuyển sang dict {start: [i,j], end: [i,j]}
    result = []
    for start, end in result_set:
        # result.append({'start': list(start), 'end': list(end)})
        result.append({ 'start': [int(start[0]), int(start[1])],'end': [int(end[0]), int(end[1])]})

    return result, min_i, max_i, min_j, max_j

# --- Dữ liệu test (bạn thay bằng mask thật) ---
mask = np.zeros((10, 15), dtype=int)
mask[1, 1:6] = 1
mask[2, 2:8] = 1
mask[5, 5:12] = 1
mask[8, 3:5] = 1
mask[8, 7] = 1  # nhiễu lẻ
print(mask)
arr_numpy = np.loadtxt("wall1.csv", skiprows=1, delimiter=",", dtype=int)
# arr_numpy=mask
straightened = smooth_map_cv2(arr_numpy, method='both', structure_size=3)
straightened = straighten_wall_map(straightened, epsilon = 0.9,min_area=50)
array_list = straightened.T.tolist()
result, min_i, max_i, min_j, max_j = find_consecutive_ranges(straightened)
print("straightened=",straightened)
plt.clf()  # xóa figure cũ
plt.imshow(straightened, cmap='gray')
plt.show()
# # --- Tìm các hình chữ nhật theo chiều ngang ---
# rects = find_horizontal_rectangles(mask, min_len=2)

# # --- Vẽ ---
# fig, ax = plt.subplots(figsize=(10, 6))

# # Hiển thị mask gốc
# ax.imshow(mask, cmap='gray_r', origin='upper', extent=(0, mask.shape[1], mask.shape[0], 0))

# # Vẽ từng hình chữ nhật tìm được
# for i, j, h, w in rects:
#     rect = plt.Rectangle((j, i), w, h, edgecolor='red', facecolor='none', linewidth=2)
#     ax.add_patch(rect)

# # Tắt trục và lưới
# ax.set_xticks([]); ax.set_yticks([])
# ax.set_xlim([0, mask.shape[1]])
# ax.set_ylim([mask.shape[0], 0])
# ax.set_aspect('equal')

# plt.title("Các hình chữ nhật dài theo hàng ngang (không trùng nhau)")
# plt.tight_layout()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# def find_full_1_rectangles(mask):
#     visited = np.zeros_like(mask, dtype=bool)
#     rectangles = []
#     rows, cols = mask.shape

#     for i in range(rows):
#         j = 0
#         while j < cols:
#             if mask[i, j] == 1 and not visited[i, j]:
#                 start_j = j
#                 while j < cols and mask[i, j] == 1 and not visited[i, j]:
#                     j += 1
#                 end_j = j

#                 height = 1
#                 while (i + height < rows and
#                        np.all(mask[i + height, start_j:end_j] == 1) and
#                        np.all(visited[i + height, start_j:end_j] == False)):
#                     height += 1

#                 width = end_j - start_j

#                 if width > 2 and height > 2:
#                     rectangles.append(((i, start_j), (i + height, end_j)))
#                     visited[i:i + height, start_j:end_j] = True
#                 else:
#                     j = end_j
#             else:
#                 j += 1

#     return rectangles

# # ===== Load mask =====
# mask = np.loadtxt("wall1.csv", skiprows=1, delimiter=",", dtype=int).astype(np.uint8)

# # ===== Lọc hình chữ nhật =====
# rects = find_full_1_rectangles(mask)

# # ===== Tạo mask mới (chỉ giữ các pixel hợp lệ) =====
# filtered_mask = np.zeros_like(mask, dtype=np.uint8)
# for (minr, minc), (maxr, maxc) in rects:
#     filtered_mask[minr:maxr, minc:maxc] = 1

# # ===== Hiển thị kết quả =====
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# axs[0].imshow(mask, cmap='gray')
# axs[0].set_title("Original Mask")

# axs[1].imshow(filtered_mask, cmap='gray')
# axs[1].set_title("Filtered Mask (valid rectangles only)")

# for ax in axs:
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.tight_layout()
# plt.show()
