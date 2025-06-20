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
    # opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    return closed

def extract_large_contours(binary_img, min_area=50):
    """Lấy contour có diện tích lớn, bỏ phần lồi nhỏ như mép cửa"""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    return large_contours
  
def group_points_axis_aligned(points, tolerance=1):
    """Gom các điểm (x, y) thành các hình chữ nhật khít trục X/Y"""
    from collections import defaultdict
    point_set = set(points)
    visited = set()
    rects = []

    for x, y in points:
        if (x, y) in visited:
            continue

        # mở rộng theo X
        x_end = x
        while (x_end + 1, y) in point_set:
            x_end += 1

        # mở rộng theo Y
        y_end = y
        while all((xi, y_end + 1) in point_set for xi in range(x, x_end + 1)):
            y_end += 1

        # Đánh dấu đã dùng
        for xi in range(x, x_end + 1):
            for yi in range(y, y_end + 1):
                visited.add((xi, yi))

        # Thêm hình chữ nhật
        rects.append((x, y, x_end - x + 1, y_end - y + 1))  # (x, y, w, h)

    return rects
# def extract_large_contours(binary_img, min_area=4):
#     """
#     Chia nhỏ ảnh nhị phân thành các contour hình chữ nhật axis-aligned (khít theo lưới).
#     Trả về list các contour hình (Nx1x2) như OpenCV yêu cầu.
#     """
#     h, w = binary_img.shape
#     visited = np.zeros_like(binary_img, dtype=bool)
#     contours = []

#     for i in range(h):
#         for j in range(w):
#             if binary_img[i, j] == 1 and not visited[i, j]:
#                 # Bắt đầu mở rộng từ (i, j)
#                 x1, y1 = j, i
#                 x2 = x1
#                 while x2 + 1 < w and binary_img[y1, x2 + 1] == 1 and not visited[y1, x2 + 1]:
#                     x2 += 1

#                 y2 = y1
#                 valid = True
#                 while y2 + 1 < h and valid:
#                     for x in range(x1, x2 + 1):
#                         if binary_img[y2 + 1, x] != 1 or visited[y2 + 1, x]:
#                             valid = False
#                             break
#                     if valid:
#                         y2 += 1

#                 # Đánh dấu đã dùng
#                 for y in range(y1, y2 + 1):
#                     for x in range(x1, x2 + 1):
#                         visited[y, x] = True

#                 # Tính diện tích và thêm contour nếu đủ lớn
#                 area = (x2 - x1 + 1) * (y2 - y1 + 1)
#                 if area >= min_area:
#                     cnt = np.array([
#                         [[x1, y1]],
#                         [[x2 + 1, y1]],
#                         [[x2 + 1, y2 + 1]],
#                         [[x1, y2 + 1]]
#                     ], dtype=np.int32)
#                     contours.append(cnt)

#     return contours

# def make_contour_axis_aligned(contour, epsilon=3):
#     """Xấp xỉ contour vuông góc và căn chỉnh theo trục X hoặc Z"""
#     approx = cv2.approxPolyDP(contour, epsilon, closed=True)
#     aligned = []

#     for pt in approx:
#         aligned.append(pt[0].copy())

#     for i in range(len(aligned) - 1):
#         p1 = aligned[i]
#         p2 = aligned[i + 1]

#         dx = abs(p1[0] - p2[0])
#         dy = abs(p1[1] - p2[1])
#         if dx > dy:
#             p2[1] = p1[1]  # Ép ngang
#         elif dy > dx:
#             p2[0] = p1[0]  # Ép dọc
#         # Nếu gần bằng nhau thì giữ nguyên (để tránh phá đường nghiêng nhẹ)

#     return np.array(aligned, dtype=np.int32).reshape(-1, 1, 2)

# def make_contour_axis_aligned(contour, epsilon=3):
#     """Xấp xỉ contour và căn chỉnh theo trục X/Y"""
#     approx = cv2.approxPolyDP(contour, epsilon, closed=True)
#     aligned = []

#     for pt in approx:
#         aligned.append(pt[0].copy())

#     for i in range(len(aligned)):
#         p1 = aligned[i]
#         p2 = aligned[(i + 1) % len(aligned)]  # nối vòng tròn

#         dx = abs(p1[0] - p2[0])
#         dy = abs(p1[1] - p2[1])

#         if dx > dy:
#             p2[1] = p1[1]
#         elif dy > dx:
#             p2[0] = p1[0]

#     # Đảm bảo khép kín
#     aligned.append(aligned[0])

#     return np.array(aligned, dtype=np.int32).reshape(-1, 1, 2)

def make_contour_axis_aligned(contour, epsilon=3, merge_thresh=2):
    """Xấp xỉ contour vuông góc, và gom các cạnh gần nhau thành thẳng hàng"""
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    aligned = [pt[0].copy() for pt in approx]

    # Bước 1: Làm vuông từng đoạn
    for i in range(len(aligned)):
        p1 = aligned[i]
        p2 = aligned[(i + 1) % len(aligned)]
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        if dx > dy:
            p2[1] = p1[1]
        elif dy > dx:
            p2[0] = p1[0]

    # Bước 2: Gom cạnh gần nhau theo trục ngắn (≤ merge_thresh)
    def merge_aligned_points(points, axis):
        coord_idx = 0 if axis == 'x' else 1
        groups = []
        used = [False] * len(points)

        for i in range(len(points)):
            if used[i]:
                continue
            group = [i]
            for j in range(i+1, len(points)):
                if not used[j] and abs(points[i][coord_idx] - points[j][coord_idx]) <= merge_thresh:
                    group.append(j)
            mean_val = int(round(np.mean([points[k][coord_idx] for k in group])))
            for k in group:
                points[k][coord_idx] = mean_val
                used[k] = True

    merge_aligned_points(aligned, axis='x')
    merge_aligned_points(aligned, axis='y')

    aligned.append(aligned[0])  # Đảm bảo khép kín
    return np.array(aligned, dtype=np.int32).reshape(-1, 1, 2)



# def make_contour_axis_aligned(contour, mask, epsilon=3):
#     """Xấp xỉ contour vuông góc, gom điểm gần nhau, và làm sạch mask theo trục"""

#     H, W = mask.shape
#     merge_thresh = 2

#     approx = cv2.approxPolyDP(contour, epsilon, closed=True)
#     aligned = [pt[0].copy() for pt in approx]

#     # Bước 1: Làm vuông từng đoạn
#     for i in range(len(aligned)):
#         p1 = aligned[i]
#         p2 = aligned[(i + 1) % len(aligned)]
#         dx = abs(p1[0] - p2[0])
#         dy = abs(p1[1] - p2[1])
#         if dx > dy:
#             p2[1] = p1[1]
#         elif dy > dx:
#             p2[0] = p1[0]

#     # Bước 2: Gom các điểm gần nhau (thẳng hàng)
#     def merge_aligned_points(points, axis):
#         coord_idx = 0 if axis == 'x' else 1
#         used = [False] * len(points)
#         for i in range(len(points)):
#             if used[i]:
#                 continue
#             group = [i]
#             for j in range(i+1, len(points)):
#                 if not used[j] and abs(points[i][coord_idx] - points[j][coord_idx]) <= merge_thresh:
#                     group.append(j)
#             mean_val = int(round(np.mean([points[k][coord_idx] for k in group])))
#             for k in group:
#                 points[k][coord_idx] = mean_val
#                 used[k] = True

#     merge_aligned_points(aligned, 'x')
#     merge_aligned_points(aligned, 'y')

#     # Bước 3: Xóa vùng nhiễu nếu trên trục số lượng nhãn 1 quá ít
#     def clean_mask_lines(mask, axis):
#         counts = {}
#         for y in range(H):
#             for x in range(W):
#                 if mask[y, x] == 1:
#                     key = x if axis == 'x' else y
#                     counts[key] = counts.get(key, 0) + 1

#         for key, count in counts.items():
#             total = H if axis == 'x' else W
#             if count / total <= 0.3:
#                 if axis == 'x' and 0 <= key < W:
#                     mask[:, key] = 0
#                 elif axis == 'y' and 0 <= key < H:
#                     mask[key, :] = 0

#     clean_mask_lines(mask, 'x')
#     clean_mask_lines(mask, 'y')

#     # Khép kín contour
#     aligned.append(aligned[0])
#     return np.array(aligned, dtype=np.int32).reshape(-1, 1, 2)



# def draw_clean_walls(shape, contours, epsilon=3, min_area=50):
#     """Vẽ lại các tường đã làm sạch và vuông hóa"""
#     canvas = np.zeros(shape, dtype=np.uint8)
#     epsilon_2=epsilon
#     for idx, cnt in enumerate(contours):
#         area = cv2.contourArea(cnt)
#         if area < min_area:
#             continue
#        # Tạo canvas hiển thị debug cho từng contour
#         # Tạo canvas hiển thị debug cho từng contour
#         aligned_cnt = make_contour_axis_aligned(cnt, epsilon_2)
#         cv2.drawContours(canvas, [aligned_cnt], -1, 255, thickness=-1)

#         # Tạo ảnh RGB hiển thị
#         overlay_orig = np.zeros((*shape, 3), dtype=np.uint8)
#         overlay_aligned = np.zeros((*shape, 3), dtype=np.uint8)

#         # Vẽ contour gốc màu xanh
#         cv2.drawContours(overlay_orig, [cnt], -1, (0, 255, 0), 1)

#         # Vẽ contour đã vuông hóa màu đỏ
#         cv2.drawContours(overlay_aligned, [aligned_cnt], -1, (255, 0, 0), 1)

#         # Hiển thị 2 ảnh cạnh nhau
#         fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#         axs[0].imshow(overlay_orig)
#         axs[0].set_title(f'Contour gốc (area={area:.1f})')
#         axs[0].axis('off')

#         axs[1].imshow(overlay_aligned)
#         axs[1].set_title('Sau vuông hóa')
#         axs[1].axis('off')

#         plt.tight_layout()
#         plt.show()
#     return (canvas > 0).astype(np.uint8)

def draw_clean_walls(shape, contours, epsilon=3, min_area=50):
    canvas = np.zeros(shape, dtype=np.uint8)
    epsilon_2 = epsilon

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Vuông hóa contour
        aligned_cnt = make_contour_axis_aligned(cnt, epsilon_2)

        # Fill polygon
        cv2.fillPoly(canvas, [aligned_cnt], 255)

    # Dilation để làm dày nét tường sau vuông hóa
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canvas = cv2.dilate(canvas, kernel, iterations=1)

    return (canvas > 0).astype(np.uint8)


def straighten_wall_map(wall_map, epsilon=3, min_area=50):
    """Pipeline xử lý toàn diện: mịn hóa + lọc nhiễu + vuông hóa"""
    processed = preprocess_wall_map(wall_map)
    # processed= smooth_map_cv2(wall_map, method='both', structure_size=3)
    # processed=(wall_map * 255).astype(np.uint8)
    contours = extract_large_contours(processed, min_area=min_area)
    
    debug_canvas = np.stack([processed * 255]*3, axis=-1).astype(np.uint8) 
    
    cv2.drawContours(debug_canvas, contours, -1, (0, 255, 0), 1)

    plt.figure(figsize=(18, 18))
    plt.imshow(debug_canvas)
    plt.title("Tất cả contours trước vuông hóa")
    plt.axis("off")
    plt.show()
    straightened = draw_clean_walls(processed.shape, contours, epsilon=epsilon, min_area=min_area)
    
    return straightened


def smooth_map_cv2(grid, method='open', structure_size=3):
    kernel = np.ones((structure_size, structure_size), np.uint8)

    if method == 'open':
        result = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    elif method == 'close':
        result = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    elif method == 'both':
        opened = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        result = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel)
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
image_path = "./resources/30939153.jpg"
# image_path = "./resources/example5.jpg"
# image_path = "./resources/123.jpg"
# image_path = "./resources/Screenshot_6.png"

# Trích xuất và lọc nhiễu
image, clean_mask = extract_black_pixels(image_path, threshold=30, min_region_size=30)
print("straightened=",image)
plt.clf()  # xóa figure cũ
plt.figure(figsize=(18, 18))
plt.imshow(clean_mask, cmap='gray')
plt.show()
arr_numpy=clean_mask
# arr_numpy=mask
straightened = smooth_map_cv2(arr_numpy, method='both', structure_size=3)
straightened = straighten_wall_map(straightened, epsilon = 1,min_area=30)
array_list = straightened.T.tolist()
result, min_i, max_i, min_j, max_j = find_consecutive_ranges(straightened)
np.savetxt("test1.csv", straightened, delimiter=",", fmt='%d')
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

def group_consecutive_regions(arr):
    """
    Group các đoạn pixel 1 liên tiếp theo hàng hoặc cột trong mảng nhị phân.

    Args:
        arr (np.ndarray): 2D array chỉ chứa 0 hoặc 1

    Returns:
        List[Dict]: mỗi dict có dạng {'start': [i, j], 'end': [i, j]}
    """
    result = []

    # Theo hàng (i cố định, j tăng)
    for i in range(arr.shape[0]):
        js = np.where(arr[i] == 1)[0]
        if len(js) == 0:
            continue
        for j_start, j_end in merge_consecutive(sorted(js)):
            result.append({'start': [i, j_start], 'end': [i, j_end]})

    # Theo cột (j cố định, i tăng)
    for j in range(arr.shape[1]):
        is_ = np.where(arr[:, j] == 1)[0]
        if len(is_) == 0:
            continue
        for i_start, i_end in merge_consecutive(sorted(is_)):
            # Tránh lặp nếu đoạn chỉ 1 điểm và đã có ở chiều hàng
            if i_start == i_end:
                continue
            result.append({'start': [i_start, j], 'end': [i_end, j]})

    return result

arr = np.array([
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
])


kq=group_consecutive_regions(arr)
print(kq)
