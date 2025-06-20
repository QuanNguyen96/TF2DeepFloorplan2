import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

# Step 1: Preprocessing
def preprocess_image(path, threshold=200):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.medianBlur(binary, 3)
    return binary

# Step 2: Thickness Filter
def thickness_filter(mask, min_thick=3):
    dist = distance_transform_edt(mask)
    mask[dist < min_thick] = 0
    return mask

# Step 3: Skeletonization
def extract_skeleton(mask):
    skeleton = skeletonize(mask > 0)
    return skeleton.astype(np.uint8) * 255

# Step 4: Filter short skeletons
def filter_skeleton(skeleton, min_len=15):
    labeled = label(skeleton)
    filtered = remove_small_objects(labeled, min_size=min_len)
    return (filtered > 0).astype(np.uint8) * 255

# Step 5: Gap Closing
def close_gaps(mask, kernel_size=7):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed

# Step 6: Extract segments and recover rectangles with thickness
def extract_wall_rects_from_skeleton(skeleton_bin, distance_map, min_len=10):
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
            if len(path) < min_len:
                continue

            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            cx, cy = int(np.mean(xs)), int(np.mean(ys))

            # Hướng đoạn (ngang hay dọc)
            horizontal = (max_x - min_x) >= (max_y - min_y)

            # Ước lượng độ dày từ skeleton vào distance_map
            thickness_vals = [distance_map[y, x] for (y, x) in path]
            avg_thickness = int(np.mean(thickness_vals) * 2)  # distance map là nửa chiều rộng

            if horizontal:
                y1 = min_y - avg_thickness // 2
                h = avg_thickness
                rect = {
                    "x": int(min_x),
                    "y": int(y1),
                    "width": int(max_x - min_x + 1),
                    "height": int(h)
                }
            else:
                x1 = min_x - avg_thickness // 2
                w = avg_thickness
                rect = {
                    "x": int(x1),
                    "y": int(min_y),
                    "width": int(w),
                    "height": int(max_y - min_y + 1)
                }

            wall_rects.append(rect)
    return wall_rects

# Step 7: Visualization
def draw_results(img, wall_rects):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for rect in wall_rects:
        x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)  # Blue
    plt.figure(figsize=(12, 8))
    plt.imshow(vis[..., ::-1])
    plt.title("Detected Wall Rectangles (Blue)")
    plt.axis('off')
    plt.show()

# Step 8: Main pipeline
def main_pipeline(path):
    orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    bin_mask = preprocess_image(path)
    thick_mask = thickness_filter(bin_mask.copy())
    distance_map = distance_transform_edt(thick_mask)
    skel = extract_skeleton(thick_mask)
    filtered = filter_skeleton(skel)
    closed = close_gaps(filtered)

    wall_rects = extract_wall_rects_from_skeleton(closed, distance_map)

    with open("wall_segments.json", "w") as f:
        json.dump(wall_rects, f, indent=2)
    print(f"✅ Saved {len(wall_rects)} wall segments to wall_segments.json")

    draw_results(orig, wall_rects)

# Example usage
img_path = "./resources/example4.png"  # hoặc ảnh khác
img_path="./resources/123.jpg"
main_pipeline(img_path)
