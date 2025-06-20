# Wall & Room Detection Pipeline with proper virtual wall-based gap closing (as per Cabrera-Vargas)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

# --- Step 1: Preprocessing ---
def preprocess_image(path, threshold=200):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.medianBlur(binary, 3)
    return binary

# --- Step 2: Wall mask filtering by thickness ---
def thickness_filter(mask, min_thick=3):
    dist = distance_transform_edt(mask)
    mask[dist < min_thick] = 0
    return mask

# --- Step 3: Skeletonize and filter small fragments ---
def extract_and_filter_skeleton(mask, min_len=15):
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    labeled = label(skeleton)
    filtered = remove_small_objects(labeled, min_size=min_len)
    return (filtered > 0).astype(np.uint8) * 255

# --- Step 4: Extract wall segments (approximate) ---
def extract_wall_segments(skeleton):
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if len(cnt) >= 2:
            x1, y1 = cnt[0][0]
            x2, y2 = cnt[-1][0]
            lines.append(((x1, y1), (x2, y2)))
    return lines

# --- Step 5: Virtual wall generation (gap closing) ---
def are_colinear(p1, p2, p3, max_deviation=10):
    # Area of triangle close to 0 means colinear
    area = abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0)
    return area < max_deviation

def connect_wall_segments(lines, image_shape, max_dist=40, thickness=2):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for i, (a1, a2) in enumerate(lines):
        for j, (b1, b2) in enumerate(lines):
            if i == j:
                continue
            for pa in [a1, a2]:
                for pb in [b1, b2]:
                    dist = np.linalg.norm(np.array(pa) - np.array(pb))
                    if dist < max_dist and are_colinear(a1, a2, pb):
                        cv2.line(mask, pa, pb, 255, thickness)
    return mask

# --- Step 6: Merge with original mask and close remaining gaps ---
def close_gaps(original_mask, virtual_mask, kernel_size=5):
    combined = cv2.bitwise_or(original_mask, virtual_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return closed

# --- Step 7: Room detection ---
def detect_rooms(wall_mask, min_area=500):
    inv = cv2.bitwise_not(wall_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    rooms = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            rooms.append((x, y, w, h))
    return rooms

# --- Step 8: Visualization ---
def draw_results(img, wall_mask, rooms):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis[wall_mask > 0] = [0, 0, 255]  # Red walls
    for (x, y, w, h) in rooms:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green room boxes
    plt.figure(figsize=(12, 8))
    plt.imshow(vis[..., ::-1])
    plt.title("Walls (Red) and Rooms (Green Boxes)")
    plt.axis('off')
    plt.show()

def draw_results_bw(wall_mask):
    wall_view = np.zeros_like(wall_mask)
    wall_view[wall_mask > 0] = 255
    plt.figure(figsize=(10, 8))
    plt.imshow(wall_view, cmap='gray')
    plt.title("Wall Mask (White = Wall)")
    plt.axis('off')
    plt.show()

# --- Main pipeline ---
def main_pipeline(path):
    orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    wall_bin = preprocess_image(path)
    filtered = thickness_filter(wall_bin.copy())
    skeleton = extract_and_filter_skeleton(filtered)
    wall_segments = extract_wall_segments(skeleton)
    virtual = connect_wall_segments(wall_segments, wall_bin.shape)
    final_mask = close_gaps(filtered, virtual)
    rooms = detect_rooms(final_mask)
    draw_results(orig, final_mask, rooms)
    draw_results_bw(final_mask)

# --- Run ---
if __name__ == "__main__":
    img_path = "./resources/example4.png"
    img_path = "./resources/30939153.jpg"
    # img_path = "./resources/Screenshot_6.png"
    main_pipeline(img_path)