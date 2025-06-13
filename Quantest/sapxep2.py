import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random

# Tạo ảnh nền trắng làm mặt bằng
floorplan_img = np.ones((300, 400)) * 255  # 300x400 pixels

# Giả lập các phòng có bounding box và loại phòng
rooms = [
    {"type": "bedroom", "bbox": (50, 50, 100, 80)},        # x, y, width, height
    {"type": "living room", "bbox": (200, 50, 150, 100)},
    {"type": "kitchen", "bbox": (50, 160, 100, 80)},
    {"type": "bathroom", "bbox": (200, 170, 80, 60)}
]

# Nội thất theo từng loại phòng
room_furniture = {
    "bedroom": ["bed", "wardrobe", "desk"],
    "living room": ["sofa", "tv", "coffee table"],
    "kitchen": ["sink", "stove", "fridge"],
    "bathroom": ["toilet", "sink", "shower"]
}

# Màu cho từng loại nội thất
furniture_colors = {
    "bed": "blue", "wardrobe": "green", "desk": "purple",
    "sofa": "orange", "tv": "black", "coffee table": "brown",
    "sink": "cyan", "stove": "red", "fridge": "gray",
    "toilet": "pink", "shower": "navy"
}

# Vẽ ảnh mặt bằng + nội thất
fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(floorplan_img, cmap='gray')

# Duyệt từng phòng để vẽ
for room in rooms:
    room_type = room["type"]
    x, y, w, h = room["bbox"]

    # Vẽ viền phòng
    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.text(x + 5, y + 5, room_type, fontsize=8, verticalalignment='top', color='black')

    # Thêm nội thất
    furniture_list = room_furniture.get(room_type.lower(), [])
    for item in furniture_list:
        # Vị trí ngẫu nhiên trong phòng
        fx = x + random.randint(5, max(5, w - 25))
        fy = y + random.randint(5, max(5, h - 25))
        fw, fh = 20, 10  # kích thước nội thất đơn giản

        # Vẽ nội thất
        furn_rect = patches.Rectangle(
            (fx, fy), fw, fh,
            linewidth=1,
            edgecolor='black',
            facecolor=furniture_colors.get(item, 'gray'),
            alpha=0.7
        )
        ax.add_patch(furn_rect)
        ax.text(fx + 2, fy + 2, item, fontsize=6, verticalalignment='top', color='white')

# Hiển thị kết quả
ax.set_title("Auto-Furnished Floorplan (Simulated)", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()
