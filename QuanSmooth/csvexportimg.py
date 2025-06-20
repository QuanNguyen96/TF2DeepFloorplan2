import numpy as np
import cv2

# Đọc CSV và bỏ dòng đầu
csv_file = "E:\Data-Quan\TF2DeepFloorplan\TF2DeepFloorplan\door_origin_wall_window.csv"
data = np.loadtxt(csv_file, delimiter=",", skiprows=1, dtype=np.uint8)

# Chuyển 0/1 thành 0/255 để ra ảnh đen trắng
img = (data * 255).astype(np.uint8)

# Lưu ảnh
cv2.imwrite("mask_output.png", img)