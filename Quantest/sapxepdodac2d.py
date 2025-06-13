import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

# ====== Màu sắc cho các nhãn =======
label_colors = {
    0: 'white',    # trống
    1: 'black',    # tường
    2: 'yellow',   # cửa
    10: 'skyblue',     # phòng khách
    11: 'lightgreen',  # phòng ngủ
    12: 'lightgray',   # phòng tắm
    13: 'plum',        # nhà vệ sinh
    20: 'blue',     # giường
    21: 'orange',   # bàn
    22: 'red',      # ghế
    23: 'purple',   # tủ
    24: 'brown',    # đèn ngủ
    25: 'pink',     # tivi
    26: 'cyan',     # bàn trang điểm
    27: 'green',    # bồn tắm / bồn cầu
}

room_furniture_constraints = {
    10: [21, 22, 25],
    11: [20, 21, 22, 23, 24, 25, 26],
    12: [27],
    13: [27]
}

special_rules = {
    20: 'corner',
    21: 'center',
    22: 'wall_or_center',
    23: 'wall',
    24: 'near_bed',
    25: 'wall',
    26: 'wall',
    27: 'wall'
}

cmap = mcolors.ListedColormap([label_colors[i] for i in sorted(label_colors)])
bounds = list(sorted(label_colors)) + [max(label_colors)+1]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

def can_place(room, x, y, w, h):
    if x + w > room.shape[1] or y + h > room.shape[0]:
        return False
    area = room[y:y+h, x:x+w]
    if np.any(area > 19) or 2 in area:
        return False
    return True

def place_furniture_with_rules(room, furniture_label, size, room_mask, room_label, placed_bed_pos):
    h, w = size
    coords = list(zip(*np.where(room_mask)))
    random.shuffle(coords)
    for y, x in coords:
        if special_rules.get(furniture_label) == 'corner':
            if y not in [1, room.shape[0]-h-1] or x not in [1, room.shape[1]-w-1]:
                continue
        elif special_rules.get(furniture_label) == 'wall':
            if y != 1 and y != room.shape[0]-h-1 and x != 1 and x != room.shape[1]-w-1:
                continue
        elif special_rules.get(furniture_label) == 'near_bed':
            if placed_bed_pos is None:
                continue
            by, bx = placed_bed_pos
            if abs(by - y) > 2 or abs(bx - x) > 2:
                continue
        if can_place(room, x, y, w, h):
            room[y:y+h, x:x+w] = furniture_label
            if furniture_label == 20:
                return (y, x)
            break
    return placed_bed_pos

def visualize_room_colored(room):
    plt.figure(figsize=(6, 6))
    plt.imshow(room, cmap=cmap, norm=norm)
    plt.grid(True, which='both', color='lightgray', linewidth=0.5)
    plt.xticks([]), plt.yticks([])
    handles = [plt.Line2D([0], [0], marker='s', color='w',
                          markerfacecolor=label_colors[i], label=f"{i}", markersize=10)
               for i in sorted(label_colors)]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Tự động bố trí nội thất theo nhãn phòng và ràng buộc")
    plt.tight_layout()
    plt.show()

room = np.ones((7, 15), dtype=int)
room[1:6,1:6] = 11   # phòng ngủ
room[1:6,6:10] = 12  # phòng tắm
room[1:6,10:14] = 13 # vệ sinh
room[6:14,2:13] = 10 # phòng khách
room[6,7] = 2        # cửa

furniture_defs = {
    20: (2,3), 21: (1,2), 22: (1,1), 23: (1,2),
    24: (1,1), 25: (1,2), 26: (1,2), 27: (1,2)
}

for label in np.unique(room):
    if label < 10:
        continue
    mask = (room == label)
    placed_bed_pos = None
    for f in room_furniture_constraints[label]:
        placed_bed_pos = place_furniture_with_rules(room, f, furniture_defs[f], mask, label, placed_bed_pos)

visualize_room_colored(room)
