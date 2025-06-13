import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import sys
import numpy as np
from scipy.ndimage import generic_filter,label
from scipy.stats import mode
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

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from dfp.utils.rgb_ind_convertor import *
from dfp.utils.util import *
from dfp.utils.legend import *
from dfp.utils.settings import *
from dfp.deploy import *



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

def update_labels_by_color(array, color_map,color_to_unified_id):
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

def mode_filter_func(values):
    return mode(values)[0][0]

def get_image_shape(newr, newcw):
    if newr is not None and hasattr(newr, 'shape') and newr.size > 0:
        return newr.shape  # trả về (m, n)
    elif newcw is not None and hasattr(newcw, 'shape') and newcw.size > 0:
        return newcw.shape
    else:
        return None  # hoặc (0, 0) tùy bạn
def export_floorplan_json(newr, newcw,color_to_unified_id,unified_labels, filename="floorplan.json",):
    # print("newr shape",newr.shape)
    # newr_2d = newr.squeeze(axis=2) 
    # print("newr_2d=",newr_2d)
    # newr_smooth = generic_filter(newr_2d, mode_filter_func, size=5, mode='nearest')
    # print("newr_smooth",newr_smooth)
    # newcw_2d = newcw.squeeze(axis=2)
    # newcw = generic_filter(newcw_2d, mode_filter_func, size=5, mode='nearest')
    output = {}
    size_img = get_image_shape(newr, newcw)
    if size_img is not None:
        # phải đảo ngược lại vì shape ở đây đi từ ngoài vào trong nếu đảo ngược lại mảng shape, 
        output = {
          "sizeImg":size_img[:2][::-1]   
        }

    updated_newr = update_labels_by_color(newr, floorplan_fuse_map,color_to_unified_id)
    updated_newcw = update_labels_by_color(newcw, floorplan_boundary_map,color_to_unified_id)

    # updated_newr_smooth = generic_filter(updated_newr, mode_filter_func, size=5, mode='nearest')
    # updated_newcw_smooth = generic_filter(updated_newcw, mode_filter_func, size=5, mode='nearest')
    # print("updated_newr_smooth",updated_newr_smooth)
    
    # print("updated_newr=",updated_newr)

    # Duyệt qua tất cả nhãn từ unified_labels
    for class_id, info in unified_labels.items():
        label_name = info["label"]
        label_points = []

        # Lấy mask điểm của nhãn class_id trong newr và newcw
        mask_r = (updated_newr == class_id)
        mask_cw = (updated_newcw == class_id)
        # mask_r = (updated_newr_smooth == class_id)
        # mask_cw = (updated_newcw_smooth == class_id)

        # Lấy tất cả điểm có nhãn đó trong newr
        if np.any(mask_r):
            label_points.extend(get_points_from_mask(mask_r))

        # Lấy tất cả điểm có nhãn đó trong newcw
        if np.any(mask_cw):
            label_points.extend(get_points_from_mask(mask_cw))

        if label_points:
            output[label_name] = [{"points": label_points}]
    return output

    


def predict_data(pil_image,filename):
  if pil_image is None:
    return
  # img_path = "./resources/30939153.jpg"
  # inp = mpimg.imread(img_path)
  # # # print(inp)
  #  inp = np.array(pil_image)

  # Tạo file tạm và lưu ảnh vào
  suffix = os.path.splitext(filename)[1]  # vd: ".png"
  if not suffix:
    suffix = ".jpg"
  with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    pil_image.save(tmp.name)
    tmp_path = tmp.name  # đường dẫn tạm thời
    
  
  args = parse_args("--tomlfile ./docs/notebook.toml".split())
  args = overwrite_args_with_toml(args)
  args.image = tmp_path

#   result = main(args)

#   plt.subplot(1, 2, 1)
#   plt.imshow(inp)
#   plt.xticks([])
#   plt.yticks([])
#   plt.subplot(1, 2, 2)
#   plt.imshow(result)
#   plt.xticks([])
#   plt.yticks([])

  model,img,shp = init(args)
  logits_cw,logits_r = predict(model,img,shp)


  logits_r = tf.image.resize(logits_r,shp[:2])
  logits_cw = tf.image.resize(logits_cw,shp[:2])
  r = convert_one_hot_to_image(logits_r)[0].numpy()
  cw = convert_one_hot_to_image(logits_cw)[0].numpy()
#   plt.subplot(1,2,1)
#   plt.imshow(r.squeeze()); plt.xticks([]); plt.yticks([])
#   plt.subplot(1,2,2)
#   plt.imshow(cw.squeeze()); plt.xticks([]); plt.yticks([])


#   r_color,cw_color = colorize(r.squeeze(),cw.squeeze())
#   plt.subplot(1,2,1)
#   plt.imshow(r_color); plt.xticks([]); plt.yticks([])
#   plt.subplot(1,2,2)
#   plt.imshow(cw_color); plt.xticks([]); plt.yticks([])


  newr,newcw = post_process(r,cw,shp)

#   plt.subplot(1,2,1)
#   plt.imshow(newr.squeeze()); plt.xticks([]); plt.yticks([])
#   plt.subplot(1,2,2)
#   plt.imshow(newcw.squeeze()); plt.xticks([]); plt.yticks([])


#   newr_color,newcw_color = colorize(newr.squeeze(),newcw.squeeze())

  
#   plt.subplot(1,2,1)
#   plt.imshow(newr_color); plt.xticks([]); plt.yticks([])
#   plt.subplot(1,2,2)
#   plt.imshow(newcw_color); plt.xticks([]); plt.yticks([])
#   plt.imshow(newr_color+newcw_color); plt.xticks([]); plt.yticks([])
# #   buf = io.BytesIO()
#   os.makedirs('./output', exist_ok=True)
#   plt.savefig(f'./output/{filename}', bbox_inches='tight', pad_inches=0)
  
  # plt.show()


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

  results = export_floorplan_json(newr, newcw,color_to_unified_id,unified_labels, filename="floorplan4.json",)
  return results
  

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


def predict_data_v2(pil_image,filename):
  if pil_image is None:
    return
  # img_path = "./resources/30939153.jpg"
  # inp = mpimg.imread(img_path)
  # # # print(inp)
  #  inp = np.array(pil_image)

  # Tạo file tạm và lưu ảnh vào
  suffix = os.path.splitext(filename)[1]  # vd: ".png"
  if not suffix:
    suffix = ".jpg"
  with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    pil_image.save(tmp.name)
    tmp_path = tmp.name  # đường dẫn tạm thời
    
  
  args = parse_args("--tomlfile ./docs/notebook.toml".split())
  args = overwrite_args_with_toml(args)
  args.image = tmp_path

#   result = main(args)

#   plt.subplot(1, 2, 1)
#   plt.imshow(inp)
#   plt.xticks([])
#   plt.yticks([])
#   plt.subplot(1, 2, 2)
#   plt.imshow(result)
#   plt.xticks([])
#   plt.yticks([])

  model,img,shp = init(args)
  logits_cw,logits_r = predict(model,img,shp)


  logits_r = tf.image.resize(logits_r,shp[:2])
  logits_cw = tf.image.resize(logits_cw,shp[:2])
  r = convert_one_hot_to_image(logits_r)[0].numpy()
  cw = convert_one_hot_to_image(logits_cw)[0].numpy()
#   plt.subplot(1,2,1)
#   plt.imshow(r.squeeze()); plt.xticks([]); plt.yticks([])
#   plt.subplot(1,2,2)
#   plt.imshow(cw.squeeze()); plt.xticks([]); plt.yticks([])


#   r_color,cw_color = colorize(r.squeeze(),cw.squeeze())
#   plt.subplot(1,2,1)
#   plt.imshow(r_color); plt.xticks([]); plt.yticks([])
#   plt.subplot(1,2,2)
#   plt.imshow(cw_color); plt.xticks([]); plt.yticks([])


  newr,newcw = post_process(r,cw,shp)

#   plt.subplot(1,2,1)
#   plt.imshow(newr.squeeze()); plt.xticks([]); plt.yticks([])
#   plt.subplot(1,2,2)
#   plt.imshow(newcw.squeeze()); plt.xticks([]); plt.yticks([])


#   newr_color,newcw_color = colorize(newr.squeeze(),newcw.squeeze())

  
#   plt.subplot(1,2,1)
#   plt.imshow(newr_color); plt.xticks([]); plt.yticks([])
#   plt.subplot(1,2,2)
#   plt.imshow(newcw_color); plt.xticks([]); plt.yticks([])
#   plt.imshow(newr_color+newcw_color); plt.xticks([]); plt.yticks([])
# #   buf = io.BytesIO()
#   os.makedirs('./output', exist_ok=True)
#   plt.savefig(f'./output/{filename}', bbox_inches='tight', pad_inches=0)
  
  # plt.show()


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

  results = export_floorplan_json(newr, newcw,color_to_unified_id,unified_labels, filename="floorplan4.json",)
#   image_path = "./resources/30939153.jpg"
  image_2, clean_mask = extract_black_pixels(tmp_path, threshold=30, min_region_size=30)
  traightened_smooth = smooth_map_cv2(clean_mask, method='both', structure_size=3)
  traightened_smooth = straighten_wall_map(traightened_smooth, epsilon = 2,min_area=50)
  ketqua = group_consecutive_regions(traightened_smooth)
#   print("ketqua=",ketqua)
#   plt.clf()  # xóa figure cũ
#   plt.imshow(ketqua, cmap='gray')
#   plt.show()
  return results
  
  
# predict_data("fsdfds")







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


# def draw_clean_walls(shape, contours, epsilon=3, min_area=50):
#     """Vẽ lại các tường đã làm sạch và vuông hóa"""
#     canvas = np.zeros(shape, dtype=np.uint8)
#     for cnt in contours:
#         if cv2.contourArea(cnt) < min_area:
#             continue
#         aligned_cnt = make_contour_axis_aligned(cnt, epsilon)
#         cv2.drawContours(canvas, [aligned_cnt], -1, 255, thickness=-1)
#     return (canvas > 0).astype(np.uint8)

def draw_clean_walls(shape, contours, epsilon=3, min_area=50):
    """Vẽ lại các tường đã làm sạch và vuông hóa"""
    canvas = np.zeros(shape, dtype=np.uint8)
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
       # Tạo canvas hiển thị debug cho từng contour
        # Tạo canvas hiển thị debug cho từng contour
        aligned_cnt = make_contour_axis_aligned(cnt, epsilon)
        cv2.drawContours(canvas, [aligned_cnt], -1, 255, thickness=-1)
    return (canvas > 0).astype(np.uint8)

# def straighten_wall_map(wall_map, epsilon=3, min_area=50):
#     """Pipeline xử lý toàn diện: mịn hóa + lọc nhiễu + vuông hóa"""
#     processed = preprocess_wall_map(wall_map)
#     contours = extract_large_contours(processed, min_area=min_area)
#     straightened = draw_clean_walls(processed.shape, contours, epsilon=epsilon, min_area=min_area)
#     return straightened
def straighten_wall_map(wall_map, epsilon=3, min_area=50):
    """Pipeline xử lý toàn diện: mịn hóa + lọc nhiễu + vuông hóa"""
    processed = preprocess_wall_map(wall_map)
    contours = extract_large_contours(processed, min_area=min_area)
    
    debug_canvas = np.stack([processed * 255]*3, axis=-1).astype(np.uint8) 
    
    cv2.drawContours(debug_canvas, contours, -1, (0, 255, 0), 1)

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


def create_region_grid(size, regions):
    """
    Tạo mảng 2D có giá trị 1 ở các vùng từ start đến end, còn lại là 0.

    Args:
        size (list or tuple): Kích thước mảng [m, n] (số hàng, số cột).
        regions (list of dict): Mỗi dict có 'start' và 'end' là 2 điểm [i, j].

    Returns:
        np.ndarray: Mảng 2D có đánh dấu các vùng bằng 1.
    """
    grid = np.zeros((size[0], size[1]), dtype=int)

    for r in regions:
        start = r['start']
        end = r['end']

        min_i = max(min(start[0], end[0]), 0)
        max_i = min(max(start[0], end[0]), size[0] - 1)

        min_j = max(min(start[1], end[1]), 0)
        max_j = min(max(start[1], end[1]), size[1] - 1)

        grid[min_i:max_i+1, min_j:max_j+1] = 1

    return grid

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

def smooth_wall(size,walls):
    arr_numpy = create_region_grid(size, walls)
    straightened = smooth_map_cv2(arr_numpy, method='both', structure_size=3)
    straightened = straighten_wall_map(straightened, epsilon = 0.96 ,min_area=50)
    array_list = straightened.T.tolist()
    result, min_i, max_i, min_j, max_j = find_consecutive_ranges(straightened)
    # print("type results=", type(result))
    return {"result":result,"array" : array_list}


from math import sqrt
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

# def group_points(points, tolerance):
#     n = len(points)
#     uf = UnionFind(n)
#     for i in range(n):
#         for j in range(i+1, n):
#             if euclidean_dist(points[i], points[j]) <= tolerance:
#                 uf.union(i, j)
#     groups = {}
#     for i, p in enumerate(points):
#         root = uf.find(i)
#         groups.setdefault(root, []).append(p)
#     return list(groups.values())
# def fill_and_clean(arr, tolerance, label_fill, min_area=None):
#     points_label = list(zip(*np.where(arr == label_fill)))
#     if not points_label:
#         return np.zeros_like(arr)
    
#     groups = group_points(points_label, tolerance)
#     result = np.zeros_like(arr)
    
#     # Nếu không truyền min_area thì mặc định dùng max(tolerance * 2, 4)
#     if min_area is None:
#         min_area = max(tolerance * 2, 4)

#     for g in groups:
#         rows = [p[0] for p in g]
#         cols = [p[1] for p in g]
#         r_min, r_max = min(rows), max(rows)
#         c_min, c_max = min(cols), max(cols)
#         area = (r_max - r_min + 1) * (c_max - c_min + 1)
#         if area >= min_area:
#             result[r_min:r_max+1, c_min:c_max+1] = label_fill
    
#     return result

def group_points_axis_aligned(points, tolerance):
    from collections import defaultdict

    point_set = set(points)
    point_idx = {p: i for i, p in enumerate(points)}
    uf = UnionFind(len(points))

    for idx, (i, j) in enumerate(points):
        for d in range(1, tolerance + 1):
            for ni, nj in [(i + d, j), (i - d, j), (i, j + d), (i, j - d)]:
                if (ni, nj) in point_set:
                    uf.union(idx, point_idx[(ni, nj)])

    groups = defaultdict(list)
    for i, p in enumerate(points):
        root = uf.find(i)
        groups[root].append(p)
    return list(groups.values())

def fill_and_clean(arr, tolerance, label_fill, min_area=None, aspect_ratio_threshold=1.0):
    points_label = list(zip(*np.where(arr == label_fill)))
    if not points_label:
        return np.zeros_like(arr)
    
    groups = group_points_axis_aligned(points_label, tolerance)
    result = np.zeros_like(arr)
    
    if min_area is None:
        min_area = max(tolerance * 2, 4)

    for g in groups:
        rows = [p[0] for p in g]
        cols = [p[1] for p in g]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        height = r_max - r_min + 1
        width = c_max - c_min + 1
        area = height * width

        # Kiểm tra điều kiện diện tích và tỉ lệ chiều dài/rộng
        longer = max(height, width)
        shorter = min(height, width)

        if area >= min_area and longer / shorter >= aspect_ratio_threshold:
            result[r_min:r_max+1, c_min:c_max+1] = label_fill

    return result


def cal_door_window(arr):
    print("type=",type(arr))
    arr_numpy= np.array(arr)
    tolerance = 2
    label_fill = 9
    aspect_ratio = 1  # Ví dụ: chiều dài phải gấp 2.5 lần chiều rộng
    result = fill_and_clean(arr_numpy, tolerance,label_fill, min_area = 100,aspect_ratio_threshold=aspect_ratio)
    # result=arr_numpy
    print("result",result)
    plt.clf()  # xóa figure cũ
    plt.imshow(result, cmap='gray')
    plt.show()
    return result.tolist()
