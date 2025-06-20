# Wall & Room Detection Pipeline inspired by
# "Wall Extraction and Room Detection for Multi-Unit Architectural Floor Plans"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from skimage.measure import label

# Step 1: Preprocessing (binarize, clean noise)
def preprocess_image(path, threshold=200):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.medianBlur(binary, 3)
    return img, binary

# Step 2: Detect wall line segments using Hough transform
def detect_line_segments(mask):
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)
    return [tuple(line[0]) for line in lines] if lines is not None else []

# Step 3: Generate wall mask (Mw) by drawing and filling from line segments
def generate_wall_mask(lines, shape, avg_thickness=5):
    mask = np.zeros(shape, dtype=np.uint8)

    for line in lines:
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length == 0:
            continue
        dir_vec = np.array([dx, dy]) / length
        normal = np.array([-dir_vec[1], dir_vec[0]])

        # Draw central line
        cv2.line(mask, (x1, y1), (x2, y2), 255, avg_thickness)

        # Expand both sides along normal
        for shift in range(1, avg_thickness):
            offset = normal * shift
            offset = np.round(offset).astype(int)
            cv2.line(mask, (x1 + offset[0], y1 + offset[1]), (x2 + offset[0], y2 + offset[1]), 255, 1)
            cv2.line(mask, (x1 - offset[0], y1 - offset[1]), (x2 - offset[0], y2 - offset[1]), 255, 1)

    return mask

# Step 4: Gap closing using virtual walls
def bridge_wall_gaps(mask, thickness=2):
    lines = detect_line_segments(mask)
    if not lines:
        return mask, np.zeros_like(mask)

    segments = lines
    lengths = [np.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in segments]
    max_len_all = max(lengths) if lengths else 1
    virtual = np.zeros_like(mask)

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            x1a, y1a, x2a, y2a = segments[i]
            x1b, y1b, x2b, y2b = segments[j]

            la = np.hypot(x2a - x1a, y2a - y1a)
            lb = np.hypot(x2b - x1b, y2b - y1b)
            longer = segments[i] if la >= lb else segments[j]
            end_a = [(x1a, y1a), (x2a, y2a)]
            end_b = [(x1b, y1b), (x2b, y2b)]
            td = (la + lb) / 40

            for pa in end_a:
                for pb in end_b:
                    dist = np.hypot(pa[0] - pb[0], pa[1] - pb[1])
                    if dist <= 2 * max(la, lb) and dist <= max_len_all:
                        if point_to_line_distance(pb, longer) <= td:
                            if is_line_empty(mask, pa, pb):
                                cv2.line(virtual, pa, pb, 255, thickness)

    return cv2.bitwise_or(mask, virtual), virtual

def point_to_line_distance(pt, line):
    x1, y1, x2, y2 = line
    px, py = pt
    num = abs((y2 - y1)*px - (x2 - x1)*py + x2*y1 - y2*x1)
    den = np.hypot(x2 - x1, y2 - y1)
    return num / (den + 1e-6)

def is_line_empty(mask, pt1, pt2):
    line_img = np.zeros_like(mask)
    cv2.line(line_img, pt1, pt2, 255, 1)
    return np.all(mask[line_img > 0] == 0)

# Step 5: Room Detection via connected empty regions
def detect_rooms(wall_mask, davg=10, tavg=3, min_area=500):
    # Step 1: Flood fill from (0,0) to mark outside
    filled = wall_mask.copy()
    h, w = filled.shape
    cv2.floodFill(filled, np.zeros((h+2, w+2), np.uint8), (0, 0), 255)

    # Step 2: Invert filled to get inside regions
    inside = cv2.bitwise_not(filled)

    # Step 3: Connected components (potential rooms)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inside, connectivity=8)
    rooms = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        mask = (labels == i).astype(np.uint8) * 255

        # Step 4: Contour vectorization
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        # Step 5: Ellipse fitting to check axis lengths
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (major, minor), angle = ellipse
            if major > davg and minor > davg / tavg:
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

# Virtual wall visualization
def show_virtual_walls(orig, wall_mask, virtual_mask):
    vis = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    vis[wall_mask > 0] = [0, 0, 255]      # Red: original & virtual walls
    vis[virtual_mask > 0] = [255, 0, 0]   # Blue: only virtual walls
    plt.figure(figsize=(12, 8))
    plt.imshow(vis[..., ::-1])
    plt.title("Virtual Walls (Blue) + Real Walls (Red)")
    plt.axis('off')
    plt.show()

# Main pipeline
def main_pipeline(path):
    orig, bin_mask = preprocess_image(path)
    lines = detect_line_segments(bin_mask)
    wall_mask = generate_wall_mask(lines, shape=bin_mask.shape)
    bridged, virtual = bridge_wall_gaps(wall_mask)
    rooms = detect_rooms(bridged)

    # Optional: Draw final wall lines
    debug_lines = np.zeros_like(bridged)
    for x1, y1, x2, y2 in lines:
        cv2.line(debug_lines, (x1, y1), (x2, y2), 255, 1)
    plt.figure(figsize=(12, 8))
    plt.imshow(debug_lines, cmap='gray')
    plt.title("Detected Wall Line Segments")
    plt.axis('off')
    plt.show()

    show_virtual_walls(orig, bridged, virtual)
    draw_results(orig, bridged, rooms)

# Example usage:
img_path = "./resources/example4.png"
main_pipeline(img_path)