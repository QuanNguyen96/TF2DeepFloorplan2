import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import math

# --- Step 1: Đọc và xử lý nhị phân ---
def preprocess_image(path, threshold=200):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.medianBlur(binary, 3)
    return binary

# --- Step 2: Skeletonize để lấy trục tường ---
def get_skeleton(mask):
    skeleton = skeletonize(mask > 0)
    return (skeleton * 255).astype(np.uint8)

# --- Step 3: Trích xuất đoạn thẳng từ skeleton ---
def extract_wall_centerlines_from_skeleton(skeleton, min_len=20, max_gap=5):
    edges = cv2.Canny(skeleton, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=min_len, maxLineGap=max_gap)
    segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append(((x1, y1), (x2, y2)))
    return segments

# --- Step 4: Tính góc đoạn thẳng ---
def compute_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

# --- Step 5: Kiểm tra xem 2 đoạn thẳng có cùng hướng/gần nhau không ---
def are_lines_similar(l1, l2, angle_thresh=10, dist_thresh=20):
    (x1, y1), (x2, y2) = l1
    (x3, y3), (x4, y4) = l2
    angle1 = compute_angle(x1, y1, x2, y2)
    angle2 = compute_angle(x3, y3, x4, y4)
    if abs(angle1 - angle2) > angle_thresh:
        return False
    dists = [np.hypot(x1 - x3, y1 - y3), np.hypot(x2 - x4, y2 - y4),
             np.hypot(x1 - x4, y1 - y4), np.hypot(x2 - x3, y2 - y3)]
    return min(dists) < dist_thresh

# --- Step 6: Gộp các đoạn cùng hướng thành đoạn dài ---
def merge_lines(segments):
    merged = []
    used = [False] * len(segments)

    for i in range(len(segments)):
        if used[i]:
            continue
        group = [segments[i]]
        used[i] = True
        for j in range(i + 1, len(segments)):
            if not used[j] and are_lines_similar(segments[i], segments[j]):
                group.append(segments[j])
                used[j] = True

        # Gộp lại các điểm và fit thành 1 đường dài
        points = []
        for (x1, y1), (x2, y2) in group:
            points.extend([(x1, y1), (x2, y2)])
        [vx, vy, x, y] = cv2.fitLine(np.array(points), cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(vx), float(vy)
        line_len = max([np.hypot(px - x, py - y) for (px, py) in points])
        new_x1 = int(x - vx * line_len)
        new_y1 = int(y - vy * line_len)
        new_x2 = int(x + vx * line_len)
        new_y2 = int(y + vy * line_len)
        merged.append(((new_x1, new_y1), (new_x2, new_y2)))

    return merged

# --- Step 7: Vẽ kết quả ---
def visualize_wall_segments(background_img, segments):
    vis = cv2.cvtColor(background_img, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in segments:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
    plt.figure(figsize=(12, 10))
    plt.imshow(vis[..., ::-1])
    plt.title("Final Wall Skeleton Segments (One per Wall)")
    plt.axis('off')
    plt.show()

# --- Step 8: Pipeline hoàn chỉnh ---
def wall_skeleton_pipeline(path):
    orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    bin_mask = preprocess_image(path)
    skeleton = get_skeleton(bin_mask)
    raw_segments = extract_wall_centerlines_from_skeleton(skeleton)
    merged_segments = merge_lines(raw_segments)
    visualize_wall_segments(orig, merged_segments)
    return merged_segments

# --- Example usage ---
img_path = "./resources/30939153.jpg"
wall_segments = wall_skeleton_pipeline(img_path)

# In ra kết quả
for i, ((x1, y1), (x2, y2)) in enumerate(wall_segments):
    print(f"Tường {i+1}: ({x1}, {y1}) → ({x2}, {y2})")
