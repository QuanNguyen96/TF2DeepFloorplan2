# Wall & Room Detection Pipeline inspired by
# "Wall Extraction and Room Detection for Multi-Unit Architectural Floor Plans"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import networkx as nx
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

# Step 1: Preprocessing (binarize, clean noise)
def preprocess_image(path, threshold=200):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.medianBlur(binary, 3)
    return binary

# Step 2: Slice Transform (not used directly here)
def slice_transform(mask):
    h_proj = np.sum(mask, axis=1)
    v_proj = np.sum(mask, axis=0)
    return h_proj, v_proj

# Step 3: Thickness Filter
def thickness_filter(mask, min_thick=3):
    dist = distance_transform_edt(mask)
    mask[dist < min_thick] = 0
    return mask

# Step 4: Angle Matrix (not directly used here)
def compute_angle_matrix(mask):
    gx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    angle = (np.arctan2(gy, gx) * 180 / np.pi) % 180
    return angle

# Step 5: Skeletonization
def extract_skeleton(mask):
    skeleton = skeletonize(mask > 0)
    return skeleton.astype(np.uint8) * 255

# Step 6: Filter short skeletons
def filter_skeleton(skeleton, min_len=15):
    labeled = label(skeleton)
    filtered = remove_small_objects(labeled, min_size=min_len)
    return (filtered > 0).astype(np.uint8) * 255

# Step 7: Gap Closing
def close_gaps(mask, kernel_size=7):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed

# Step 8: Room Detection
def detect_rooms(wall_mask, min_area=500):
    inv = cv2.bitwise_not(wall_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    rooms = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            rooms.append((x, y, w, h))
    return rooms

# New: Step 9 - Extract wall segments (rectangular) from skeleton
def extract_wall_segments_from_skeleton(skeleton_bin):
    coords = np.argwhere(skeleton_bin > 0)
    G = nx.Graph()
    H, W = skeleton_bin.shape

    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx_ < W and skeleton_bin[ny, nx_] > 0:
                    G.add_edge((y, x), (ny, nx_))

    junctions = [n for n in G.nodes if G.degree[n] != 2]
    visited_edges = set()
    wall_rects = []

    def walk_path(start, neighbor):
        path = [start, neighbor]
        visited_edges.add(frozenset((start, neighbor)))
        prev = start
        curr = neighbor

        while True:
            nbrs = [n for n in G.neighbors(curr) if n != prev and frozenset((curr, n)) not in visited_edges]
            if not nbrs:
                break
            next_node = nbrs[0]
            path.append(next_node)
            visited_edges.add(frozenset((curr, next_node)))
            prev, curr = curr, next_node
            if G.degree[curr] != 2:
                break
        return path

    for junc in junctions:
        for nbr in G.neighbors(junc):
            if frozenset((junc, nbr)) in visited_edges:
                continue
            path = walk_path(junc, nbr)
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            rect = {
                "x": int(min_x),
                "y": int(min_y),
                "width": int(max_x - min_x + 1),
                "height": int(max_y - min_y + 1)
            }
            wall_rects.append(rect)
    return wall_rects

# Visualization
def draw_results(img, wall_mask, rooms, wall_rects=None):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis[wall_mask > 0] = [0, 0, 255]  # Red walls

    for (x, y, w, h) in rooms:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rooms

    if wall_rects:
        for rect in wall_rects:
            cv2.rectangle(vis, (rect["x"], rect["y"]),
                          (rect["x"] + rect["width"], rect["y"] + rect["height"]),
                          (255, 0, 0), 1)  # Blue rectangles

    plt.figure(figsize=(12, 8))
    plt.imshow(vis[..., ::-1])
    plt.title("Walls (Red), Rooms (Green), Wall Segments (Blue)")
    plt.axis('off')
    plt.show()

# Main pipeline
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

    wall_segments = extract_wall_segments_from_skeleton(filtered)

    with open("wall_segments.json", "w") as f:
        json.dump(wall_segments, f, indent=2)

    print(f"✅ Extracted {len(wall_segments)} wall rectangles saved to wall_segments.json")

    draw_results(orig, closed, rooms, wall_segments)

# Example usage
img_path = "./resources/example4.png"
main_pipeline(img_path)
