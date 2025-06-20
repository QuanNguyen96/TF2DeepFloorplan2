import tensorflow as tf
import sys
from dfp.net import *
from dfp.data import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import groupby
from argparse import Namespace
import os
import gc
import cv2
import json
import numpy as np
from scipy.ndimage import generic_filter
from scipy.stats import mode
import pandas  as pd

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from dfp.utils.rgb_ind_convertor import *
from dfp.utils.util import *
from dfp.utils.legend import *
from dfp.utils.settings import *
from dfp.deploy import *



# img_path = "./resources/123.jpg"
# img_path = "./resources/example4.png"
img_path = "./resources/30939153.jpg"

inp = mpimg.imread(img_path)
args = parse_args("--tomlfile ./docs/notebook.toml".split())
args = overwrite_args_with_toml(args)
args.image = img_path

result = main(args)
model,img,shp = init(args)
logits_cw,logits_r = predict(model,img,shp)

logits_r = tf.image.resize(logits_r,shp[:2])
logits_cw = tf.image.resize(logits_cw,shp[:2])
r = convert_one_hot_to_image(logits_r)[0].numpy()
cw = convert_one_hot_to_image(logits_cw)[0].numpy()
r_color,cw_color = colorize(r.squeeze(),cw.squeeze())

newr,newcw = post_process(r,cw,shp)
plt.imshow(newr.squeeze()); plt.xticks([]); plt.yticks([])


unique_labels = np.unique(newcw)
# print("Các nhãn có trong newcw:", unique_labels)

newr_color,newcw_color = colorize(newr.squeeze(),newcw.squeeze())




over255 = lambda x: [p/255 for p in x]
colors2 = [over255(rgb) for rgb in list(floorplan_fuse_map.values())]
colors = ["background", "closet", "bathroom",
          "living room\nkitchen\ndining room",
          "bedroom","hall","balcony","not used","not used",
          "door/window","wall"]
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f("s", colors2[i]) for i in range(len(colors))]
labels = colors
legend = plt.legend(handles, labels, loc=3,framealpha=1, frameon=True)


# Unified label definition
unified_labels = {
    0: {"label": "background", "color": np.array([0, 0, 0])},
    1: {"label": "closet", "color": np.array([192, 192, 224])},
    2: {"label": "bathroom", "color": np.array([192, 255, 255])},
    3: {"label": "living room/kitchen/dining room", "color": np.array([224, 255, 192])},
    4: {"label": "bedroom", "color": np.array([255, 224, 128])},
    5: {"label": "hall", "color": np.array([255, 160, 96])},
    6: {"label": "balcony", "color": np.array([255, 224, 224])},
    7: {"label": "notuse7", "color": np.array([224, 224, 224])},
    8: {"label": "notuse8", "color": np.array([224, 224, 128])},
    9: {"label": "door/window", "color": np.array([255, 60, 128])},
    10: {"label": "wall", "color": np.array([255, 255, 255])}
}

# Build color-to-ID mapping to reverse lookup
color_to_unified_id = {}
for k, v in unified_labels.items():
    color_to_unified_id[tuple(v["color"])] = k

def extract_contours(array, class_id):
    mask = (array.squeeze().astype(np.int32) == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contours_to_points(contours):
    points_list = []
    for cnt in contours:
        if cnt.shape[0] >= 3:
            pts = cnt.squeeze()
            if pts.ndim == 1:
                pts = pts[np.newaxis, :]
            points_list.append(pts.tolist())
    return points_list

def update_labels_by_color(array, color_map):
    array = array.squeeze()  # đảm bảo 2D
    h, w = array.shape
    updated = np.zeros_like(array)
    for i in range(h):
        for j in range(w):
            class_id = array[i, j]
            if class_id in color_map:
                rgb = tuple(color_map[class_id])
                new_id = color_to_unified_id.get(rgb, 0)
                updated[i, j] = new_id
    return updated

def get_points_from_mask(mask):
    # mask là mảng bool hoặc 0/1 cùng kích thước với ảnh
    # trả về list điểm [ [x1,y1], [x2,y2], ... ]
    ys, xs = np.where(mask)
    points = list(zip(xs.tolist(), ys.tolist()))  # Lưu ý thứ tự (x,y) = (col, row)
    return points

def get_image_shape(newr, newcw):
    if newr is not None and hasattr(newr, 'shape') and newr.size > 0:
        return newr.shape  # trả về (m, n)
    elif newcw is not None and hasattr(newcw, 'shape') and newcw.size > 0:
        return newcw.shape
    else:
        return None  # hoặc (0, 0) tùy bạn
    
def mode_filter_func(values):
    return mode(values)[0][0]


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
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        aligned_cnt = make_contour_axis_aligned(cnt, epsilon)
        cv2.drawContours(canvas, [aligned_cnt], -1, 255, thickness=-1)
    return (canvas > 0).astype(np.uint8)

def straighten_wall_map(wall_map, epsilon=3, min_area=50):
    """Pipeline xử lý toàn diện: mịn hóa + lọc nhiễu + vuông hóa"""
    processed = preprocess_wall_map(wall_map)
    contours = extract_large_contours(processed, min_area=min_area)
    straightened = draw_clean_walls(processed.shape, contours, epsilon=epsilon, min_area=min_area)
    return straightened



from scipy.signal import convolve2d
def smooth_labels(grid, threshold=0.5):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])  # 8-neighbor

    # Đếm số lượng hàng xóm có nhãn là 1
    neighbor_wall_count = convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=0)

    # Tổng số hàng xóm là 8, nên tỉ lệ là count / 8
    smoothed = (neighbor_wall_count / 8) >= threshold

    # Kết hợp với giá trị gốc nếu muốn giữ tường mạnh hơn
    return smoothed.astype(int)


from scipy.ndimage import binary_opening, binary_closing

def smooth_map(grid, method='open', structure_size=3):
    # Tạo kernel (structuring element)
    structure = np.ones((structure_size, structure_size))

    if method == 'open':
        # Xoá nhiễu nhỏ (wall lẻ)
        result = binary_opening(grid, structure=structure)
    elif method == 'close':
        # Lấp lỗ trống nhỏ trong wall
        result = binary_closing(grid, structure=structure)
    elif method == 'both':
        # Kết hợp cả hai: mở rồi đóng
        result = binary_closing(binary_opening(grid, structure=structure), structure=structure)
    else:
        raise ValueError("method phải là 'open', 'close' hoặc 'both'")

    return result.astype(int)


import cv2
from math import sqrt
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




class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[py] = px

def euclidean_dist(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def group_points(points, tolerance):
    n = len(points)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i+1, n):
            if euclidean_dist(points[i], points[j]) <= tolerance:
                uf.union(i, j)
    groups = {}
    for i, p in enumerate(points):
        root = uf.find(i)
        groups.setdefault(root, []).append(p)
    return list(groups.values())

def fill_and_clean(arr, tolerance, label_fill, min_area=None):
    points_label = list(zip(*np.where(arr == label_fill)))
    if not points_label:
        return np.zeros_like(arr)
    
    groups = group_points(points_label, tolerance)
    result = np.zeros_like(arr)
    
    # Nếu không truyền min_area thì mặc định dùng max(tolerance * 2, 4)
    if min_area is None:
        min_area = max(tolerance * 2, 4)

    for g in groups:
        rows = [p[0] for p in g]
        cols = [p[1] for p in g]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        area = (r_max - r_min + 1) * (c_max - c_min + 1)
        if area >= min_area:
            result[r_min:r_max+1, c_min:c_max+1] = label_fill
    
    return result


def export_floorplan_json(newr, newcw, filename="floorplan.json"):
    
    output = {}
    size_img = get_image_shape(newr, newcw)
    if size_img is not None:
        # phải đảo ngược lại vì shape ở đây đi từ ngoài vào trong nếu đảo ngược lại mảng shape,
        output = {
          "sizeImg":size_img[:2][::-1]   
        }
    # print("output",output)

    updated_newr = update_labels_by_color(newr, floorplan_fuse_map)
    updated_newcw = update_labels_by_color(newcw, floorplan_boundary_map)
    # updated_newr_smooth = generic_filter(updated_newr, mode_filter_func, size=5, mode='nearest')
    # updated_newcw_smooth = generic_filter(updated_newcw, mode_filter_func, size=5, mode='nearest')
    
    
 

    df = pd.DataFrame(updated_newcw)
    
    df_numpy=df.to_numpy()
    tolerance = 10
    label_fill = 10
    result2 = fill_and_clean(df_numpy, tolerance,label_fill, min_area = 200)
    plt.clf()  # xóa figure cũ
    plt.imshow(result2, cmap='gray')
    plt.show()
    print(result2)
    
    # df = df.astype(int)
    # tes tới nhãn nào thì để index vào df
    # df_test = 10 # đây là wall 
    df_test = 9 # đây là cửa/cửa sổ 
    df_result = pd.DataFrame(np.where(df == df_test, 1, 0))
    
    arr_numpy=df_result.to_numpy()
    df_result.to_csv("door.csv", index=False)
    straightened=arr_numpy
      # Hiển thị kết quả
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Before (Raw Prediction)")
    plt.imshow(arr_numpy, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("After (Smoothed + Aligned)")
    plt.imshow(straightened, cmap='gray')

    plt.tight_layout()
    plt.show()
    

    # Duyệt qua tất cả nhãn từ unified_labels
    for class_id, info in unified_labels.items():
        label_name = info["label"]
        label_points = []

        # Lấy mask điểm của nhãn class_id trong newr và newcw
        mask_r = (updated_newr == class_id)
        mask_cw = (updated_newcw == class_id)
        
       
        
        # Lấy tất cả điểm có nhãn đó trong newr
        if np.any(mask_r):
            label_points.extend(get_points_from_mask(mask_r))

        # Lấy tất cả điểm có nhãn đó trong newcw
        if np.any(mask_cw):
            label_points.extend(get_points_from_mask(mask_cw))

        if label_points:
            output[label_name] = [{"points": label_points}]

        # print("label_points=",label_points)
        # print("typeOf", type(label_points))
    # print("outputoutput=", output["wall"])
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Đã xuất file: {filename}")

    
# Gọi hàm với newr, newcw đã được tính toán
export_floorplan_json(newr, newcw, filename="floorplan4.json")
