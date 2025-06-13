import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from PIL import Image

def draw_bounding_boxes_on_image(img_np):
    colors, counts = np.unique(img_np.reshape(-1, img_np.shape[2]), axis=0, return_counts=True)
    print(f"Phát hiện {len(colors)} màu khác nhau trong ảnh")
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img_np)
    
    box_colors = plt.cm.get_cmap('hsv', len(colors))
    
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]], dtype=bool)
    
    for i, color in enumerate(colors):
        mask = np.all(img_np == color, axis=2)
        labeled, num_features = label(mask, structure=structure)
        
        for region_label in range(1, num_features + 1):
            coords = np.argwhere(labeled == region_label)
            if coords.size == 0:
                continue
            rmin, rmax = coords[:,0].min(), coords[:,0].max()
            cmin, cmax = coords[:,1].min(), coords[:,1].max()
            
            width = cmax - cmin
            height = rmax - rmin
            
            rect = plt.Rectangle((cmin, rmin), width, height,
                                 edgecolor=box_colors(i), facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(cmin, rmin - 5, f'{i+1}', color=box_colors(i), fontsize=12, weight='bold')
    
    ax.set_axis_off()
    plt.show()

# --- Đọc ảnh từ file ---
def process_image_file(path):
    img = Image.open(path).convert('RGB')  # Đọc ảnh, chuyển sang RGB nếu ảnh có alpha hoặc grayscale
    img_np = np.array(img)
    draw_bounding_boxes_on_image(img_np)

# Ví dụ bạn cung cấp đường dẫn ảnh:
process_image_file('./output/30939153.jpg')
