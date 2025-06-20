import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt, convolve
from skimage.draw import line as draw_line

# === STEP 1: Preprocess ===
def preprocess_image(img_path, bin_thresh=220, min_obj_size=100):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY_INV)
    binary = binary.astype(bool)
    cleaned = remove_small_objects(binary, min_size=min_obj_size)
    return cleaned

# === STEP 2: Skeletonize ===
def skeletonize_image(binary_img):
    return skeletonize(binary_img)

# === STEP 3: Distance transform ===
def get_distance_map(binary_img):
    return distance_transform_edt(binary_img)

# === STEP 4: Thickness-based wall filter ===
def filter_by_thickness(distance_map, min_thick=2, max_thick=15):
    wall_mask = (distance_map >= min_thick) & (distance_map <= max_thick)
    return wall_mask

# === STEP 5: Find endpoints in skeleton ===
def find_endpoints(skeleton):
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])
    conv = convolve(skeleton.astype(int), kernel, mode='constant')
    endpoints = np.logical_or(conv == 11, conv == 12)  # pixel with one neighbor
    coords = np.argwhere(endpoints)
    return coords

# === STEP 6: Virtual gap closing ===
def gap_closing_from_skeleton(skeleton, max_gap=15):
    endpoints = find_endpoints(skeleton)
    closed = skeleton.copy()
    for i, p1 in enumerate(endpoints):
        for j, p2 in enumerate(endpoints):
            if i >= j:
                continue
            dist = np.linalg.norm(p1 - p2)
            if dist < max_gap:
                y1, x1 = p1
                y2, x2 = p2
                rr, cc = draw_line(y1, x1, y2, x2)
                closed[rr, cc] = 1
    return closed

# === STEP 7: Vectorize wall segments ===
def extract_wall_segments(mask):
    labeled = label(mask)
    segments = []
    for region in regionprops(labeled):
        if region.area > 30:
            y0, x0, y1, x1 = *region.bbox[:2], *region.bbox[2:]
            segments.append(((x0, y0), (x1, y1)))
    return segments

# === Visualization ===
def visualize_all(binary, skeleton, distance, wall_mask, skeleton_closed):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(binary, cmap='gray'); axs[0].set_title("Binary")
    axs[1].imshow(skeleton, cmap='gray'); axs[1].set_title("Skeleton")
    axs[2].imshow(distance, cmap='viridis'); axs[2].set_title("Distance Map")
    axs[3].imshow(wall_mask, cmap='gray'); axs[3].set_title("Wall Mask")
    axs[4].imshow(skeleton_closed, cmap='gray'); axs[4].set_title("Gap Closed")
    for ax in axs: ax.axis('off')
    plt.tight_layout(); plt.show()

# === Main pipeline ===
def run_pipeline(image_path):
    binary = preprocess_image(image_path)
    skeleton = skeletonize_image(binary)
    distance = get_distance_map(binary)
    wall_mask = filter_by_thickness(distance)
    skeleton_closed = gap_closing_from_skeleton(skeleton, max_gap=15)
    wall_segments = extract_wall_segments(skeleton_closed)
    visualize_all(binary, skeleton, distance, wall_mask, skeleton_closed)
    return wall_segments

# === Example usage ===
if __name__ == "__main__":
    image_path = "your_floorplan.jpg"  # Replace with your image path
    image_path = "./resources/example4.png"
    segments = run_pipeline(image_path)
    print("Detected wall segments:")
    for seg in segments[:10]:
        print(seg)
