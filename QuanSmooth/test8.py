# Wall & Room Detection Pipeline inspired by
# "Wall Extraction and Room Detection for Multi-Unit Architectural Floor Plans"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

# Step 1: Preprocessing (binarize, clean noise)
def preprocess_image(path, threshold=200):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.medianBlur(binary, 3)
    return binary

# Step 2: Slice Transform
# Horizontal & vertical projections

def slice_transform(mask):
    h_proj = np.sum(mask, axis=1)
    v_proj = np.sum(mask, axis=0)
    return h_proj, v_proj

# Step 3: Thickness Filter

def thickness_filter(mask, min_thick=3):
    dist = distance_transform_edt(mask)
    mask[dist < min_thick] = 0
    return mask

# Step 4: Angle Matrix (Gradient direction map)

def compute_angle_matrix(mask):
    gx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    angle = (np.arctan2(gy, gx) * 180 / np.pi) % 180
    return angle

# Step 5: Skeleton & Graph

def extract_skeleton(mask):
    skeleton = skeletonize(mask > 0)
    return skeleton.astype(np.uint8) * 255

# Step 6: Geometric filtering (remove small wall segments)

def filter_skeleton(skeleton, min_len=15):
    labeled = label(skeleton)
    filtered = remove_small_objects(labeled, min_size=min_len)
    return (filtered > 0).astype(np.uint8) * 255

# Step 7: Gap Closing (Morphological or connect-close)

def close_gaps(mask, kernel_size=7):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed

# Step 8: Region Detection (room candidates)

def detect_rooms(wall_mask, min_area=500):
    inv = cv2.bitwise_not(wall_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    rooms = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            rooms.append((x, y, w, h))
    return rooms

# Visualization

def draw_results(img, wall_mask, rooms):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis[wall_mask > 0] = [0, 0, 255]
    for (x, y, w, h) in rooms:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.figure(figsize=(12, 8))
    plt.imshow(vis[..., ::-1])
    plt.title("Walls (Red) and Rooms (Green Boxes)")
    plt.axis('off')
    plt.show()

# Main full pipeline

def main_pipeline(path):
    orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    bin_mask = preprocess_image(path)
    h_proj, v_proj = slice_transform(bin_mask)
    thick_mask = thickness_filter(bin_mask.copy())
    angle_map = compute_angle_matrix(thick_mask)
    skel = extract_skeleton(thick_mask)
    filtered = filter_skeleton(skel)
    closed = close_gaps(filtered)
    rooms = detect_rooms(closed)
    draw_results(orig, closed, rooms)

# Example usage:
# img_path="./resources/123.jpg"
img_path="./resources/example4.png"
img_path = "./resources/30939153.jpg"
# img_path="./mask_output.png"
# img_path = "./resources/Screenshot_6.png"
main_pipeline(img_path)