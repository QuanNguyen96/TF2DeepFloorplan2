import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_rectangles(matrix, label_as=1):
    rows, cols = matrix.shape
    visited = np.zeros_like(matrix, dtype=bool)
    rectangles = []

    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == label_as and not visited[i, j]:
                max_col = j
                while (max_col + 1 < cols and
                       matrix[i, max_col + 1] == label_as and
                       not visited[i, max_col + 1]):
                    max_col += 1

                row_end = i
                can_expand_down = True
                while can_expand_down and row_end + 1 < rows:
                    for k in range(j, max_col + 1):
                        if matrix[row_end + 1, k] != label_as or visited[row_end + 1, k]:
                            can_expand_down = False
                            break
                    if can_expand_down:
                        row_end += 1

                visited[i:row_end+1, j:max_col+1] = True
                rectangles.append((j, i, max_col - j + 1, row_end - i + 1))

    return rectangles

def draw_rectangles(matrix, rectangles, color=(255, 255, 255), thickness=1):
    h, w = matrix.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for rect in rectangles:
        x, y, width, height = rect
        pt1 = (x, y)
        pt2 = (x + width - 1, y)
        pt3 = (x + width - 1, y + height - 1)
        pt4 = (x, y + height - 1)

        cv2.line(canvas, pt1, pt2, color, thickness)
        cv2.line(canvas, pt2, pt3, color, thickness)
        cv2.line(canvas, pt3, pt4, color, thickness)
        cv2.line(canvas, pt4, pt1, color, thickness)

    return canvas

# Ví dụ
matrix = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,0,0,0,0,0,0,0,0,1,0],
    [0,1,0,0,0,0,0,0,0,0,1,0],
    [0,1,0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,1,1,1,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
])
# matrix = np.zeros((20, 20), dtype=np.uint8)

# # Vẽ vài hình chữ nhật bằng giá trị 1
# matrix[1:5, 2:6] = 1       # HCN 4x4 ở gần góc trên trái
# matrix[6:10, 10:17] = 1    # HCN 4x7 ở giữa
# matrix[13:18, 3:8] = 1     # HCN 5x5 phía dưới trái
# matrix[12:15, 15:19] = 1   # HCN 3x4 ở gần góc dưới phải

rects = find_rectangles(matrix)
canvas = draw_rectangles(matrix, rects)
print("rects",rects)

# Hiển thị bằng matplotlib
canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(6, 6))
plt.imshow(canvas_rgb)
plt.title("Detected Rectangles")
plt.axis("off")
plt.show()

