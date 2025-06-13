import numpy as np
import cv2
import matplotlib.pyplot as plt

def greedy_rectangles(mask):
    H, W = mask.shape
    used = np.zeros_like(mask, dtype=bool)
    rectangles = []

    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and not used[i, j]:
                # Mở rộng tối đa theo OX
                max_w = 0
                while j + max_w < W and mask[i, j + max_w] == 1 and not used[i, j + max_w]:
                    max_w += 1
                # Mở rộng tối đa theo OY
                max_h = 0
                while i + max_h < H and mask[i + max_h, j] == 1 and not used[i + max_h, j]:
                    max_h += 1

                # Tìm chiều rộng tối đa (ngang)
                width = 0
                for w in range(1, max_w + 1):
                    valid = True
                    for h in range(max_h):
                        if j + w - 1 >= W or i + h >= H or mask[i + h, j + w - 1] != 1 or used[i + h, j + w - 1]:
                            valid = False
                            break
                    if valid:
                        width = w
                    else:
                        break

                # Tìm chiều cao tối đa (dọc)
                height = 0
                for h in range(1, max_h + 1):
                    valid = True
                    for w in range(max_w):
                        if i + h - 1 >= H or j + w >= W or mask[i + h - 1, j + w] != 1 or used[i + h - 1, j + w]:
                            valid = False
                            break
                    if valid:
                        height = h
                    else:
                        break

                # So sánh theo diện tích (ưu tiên hình to hơn)
                area_row = width * max_h
                area_col = height * max_w

                if area_row >= area_col:
                    rect_w, rect_h = width, max_h
                else:
                    rect_w, rect_h = max_w, height

                # Đánh dấu vùng đã xử lý
                used[i:i + rect_h, j:j + rect_w] = True
                rectangles.append((j, i, rect_w, rect_h))  # (x, y, w, h)

    return rectangles


def draw_rectangles_on_canvas(shape, rectangles):
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for idx, (x, y, w, h) in enumerate(rectangles):
        color = tuple(np.random.randint(100, 255, size=3).tolist())
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), color, 1)
    return canvas


def main():
    # Load từ CSV nhị phân (giá trị 0 và 1)
    mask = np.loadtxt("wall1.csv", delimiter=",", dtype=np.uint8)

    # Tìm các hình chữ nhật bao trọn
    rects = greedy_rectangles(mask)

    # Vẽ hình chữ nhật lên canvas
    canvas = draw_rectangles_on_canvas(mask.shape, rects)

    # Hiển thị
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Mask")
    plt.imshow(mask, cmap='gray', vmin=0, vmax=1)

    plt.subplot(1, 2, 2)
    plt.title("Greedy Rectangles")
    plt.imshow(canvas[..., ::-1])  # BGR to RGB

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
