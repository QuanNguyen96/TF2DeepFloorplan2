# Wall & Room Detection Pipeline - Exact Structure Reproduction per Cabrera-Vargas

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
from scipy.ndimage import distance_transform_edt, gaussian_filter

# --- Step 1: Preprocessing ---
def preprocess_image(path, threshold=200):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.medianBlur(binary, 3)
    return binary

# --- Step 2: Slice Transform in 4 directions ---
def slice_transform(binary):
    h, w = binary.shape
    slices = np.zeros((h, w, 4), dtype=np.uint16)
    for y in range(h):
        run = 0
        for x in range(w):
            run = run + 1 if binary[y, x] else 0
            slices[y, x, 0] = run if binary[y, x] else 0
        for x in reversed(range(w)):
            if binary[y, x]:
                slices[y, x, 0] = max(slices[y, x, 0], slices[y, x + 1, 0] if x+1<w else 0)
    for x in range(w):
        run = 0
        for y in range(h):
            run = run + 1 if binary[y, x] else 0
            slices[y, x, 1] = run if binary[y, x] else 0
        for y in reversed(range(h)):
            if binary[y, x]:
                slices[y, x, 1] = max(slices[y, x, 1], slices[y + 1, x, 1] if y+1<h else 0)
    for y in range(h):
        for x in range(w):
            if binary[y, x]:
                dy, dx = y-1, x-1
                slices[y, x, 2] = slices[dy, dx, 2] + 1 if 0 <= dy < h and 0 <= dx < w else 1
    for y in range(h):
        for x in reversed(range(w)):
            if binary[y, x]:
                dy, dx = y-1, x+1
                slices[y, x, 3] = slices[dy, dx, 3] + 1 if 0 <= dy < h and 0 <= dx < w else 1
    return slices

# --- Step 3: Slice Thickness Filter ---
def thickness_filter(slices, min_thick=3):
    return (np.min(slices[:, :, :2], axis=2) >= min_thick).astype(np.uint8) * 255

# --- Step 4: Orientation Map ---
def compute_orientation_map(slices):
    dir_idx = slices.argmax(axis=2)
    angles = np.zeros_like(dir_idx, dtype=np.uint8)
    angles[dir_idx == 0] = 0
    angles[dir_idx == 1] = 90
    angles[dir_idx == 2] = 45
    angles[dir_idx == 3] = 135
    return angles

# --- Step 5: Conditional Blur ---
def conditional_blur(mask, angle_map):
    blurred = np.zeros(mask.shape, dtype=np.float32)
    for angle in [0, 45, 90, 135]:
        sigma = 1
        region = (angle_map == angle).astype(np.uint8) * mask
        blurred += gaussian_filter(region.astype(np.float32), sigma=sigma)
    return (blurred > 127).astype(np.uint8) * 255

# --- Step 6: Skeletonization & Filtering ---
def extract_filtered_skeleton(mask, min_len=15):
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    labeled = label(skeleton)
    return (remove_small_objects(labeled, min_size=min_len) > 0).astype(np.uint8) * 255

# --- Step 7: Estimate wall thickness ---
def estimate_wall_thickness(mask):
    dist = distance_transform_edt(mask)
    return int(np.clip(np.mean(dist[dist > 0]) * 2, 4, 20))

# --- Step 8: Virtual wall connection using geometric constraints ---
def fit_wall_segments(wall_mask):
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 20]
    return rects

def connect_rects(rects, td=8):
    canvas = np.zeros((2000, 2000), dtype=np.uint8)
    for i in range(len(rects)):
        for j in range(i+1, len(rects)):
            (x1, y1, w1, h1) = rects[i]
            (x2, y2, w2, h2) = rects[j]
            cx1, cy1 = x1 + w1//2, y1 + h1//2
            cx2, cy2 = x2 + w2//2, y2 + h2//2
            dx, dy = abs(cx1 - cx2), abs(cy1 - cy2)
            if (dx < td * 2 and abs(h1 - h2) < td) or (dy < td * 2 and abs(w1 - w2) < td):
                cv2.rectangle(canvas, (min(cx1, cx2) - td//2, min(cy1, cy2) - td//2),
                              (max(cx1, cx2) + td//2, max(cy1, cy2) + td//2), 255, -1)
    return canvas

# --- Step 9: Room Detection ---
def detect_rooms(mask, min_area=500):
    inv = cv2.bitwise_not(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv)
    return [(x, y, w, h) for i, (x, y, w, h, area) in enumerate(stats) if i > 0 and area >= min_area]

# --- Visualization ---
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

# --- Pipeline ---
def main_pipeline(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    binary = preprocess_image(path)
    slices = slice_transform(binary)
    thick_mask = thickness_filter(slices)
    angle_map = compute_orientation_map(slices)
    blurred = conditional_blur(thick_mask, angle_map)
    skeleton = extract_filtered_skeleton(blurred)
    td = estimate_wall_thickness(thick_mask)
    wall_rects = fit_wall_segments(thick_mask)
    virtual_walls = connect_rects(wall_rects, td)
    final_mask = cv2.bitwise_or(thick_mask, virtual_walls)
    rooms = detect_rooms(final_mask)
    draw_results(img, final_mask, rooms)

if __name__ == "__main__":
    img_path = "./resources/example4.png"
    main_pipeline(img_path)