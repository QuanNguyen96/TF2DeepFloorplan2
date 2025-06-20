import ezdxf
import math
import matplotlib.pyplot as plt
from collections import defaultdict

# def is_wall_like(line, min_length=500, angle_tolerance=10):
#     start = line.dxf.start
#     end = line.dxf.end

#     dx = end.x - start.x
#     dy = end.y - start.y
#     length = math.hypot(dx, dy)

#     if length < min_length:
#         return False

#     angle = math.degrees(math.atan2(dy, dx)) % 180  # chỉ cần 0–180 độ

#     # Kiểm tra gần thẳng đứng hoặc ngang
#     return any(abs(angle - target) < angle_tolerance for target in [0, 90])

# def detect_wall_lines(dxf_path):
#     doc = ezdxf.readfile(dxf_path)
#     msp = doc.modelspace()
#     walls = []

#     for e in msp:
#         if e.dxftype() == 'LINE':
#             if is_wall_like(e):
#                 walls.append(((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)))

#     return walls

# # Dùng thử:
# wall_lines = detect_wall_lines("E:\Data-Quan\TF2DeepFloorplan\TF2DeepFloorplan\Quantest\demo-2.dxf")
# for line in wall_lines:
#     print(f"Tường: {line[0]} → {line[1]}")


# def summarize_labels_to_layers(dxf_path):
#     doc = ezdxf.readfile(dxf_path)
#     msp = doc.modelspace()

#     # Lưu: nhãn → tập hợp các layer chứa nó
#     label_layers = defaultdict(set)

#     for e in msp:
#         if e.dxftype() == 'TEXT':
#             text = e.dxf.text.strip()
#             layer = e.dxf.layer
#             label_layers[text].add(layer)
#         elif e.dxftype() == 'MTEXT':
#             text = e.text.strip()
#             layer = e.dxf.layer
#             label_layers[text].add(layer)

#     # In kết quả
#     for label, layers in label_layers.items():
#         print(f"🏷️ Nhãn: \"{label}\"")
#         for layer in layers:
#             print(f"  📂 Layer: {layer}")
#         print("-" * 40)

# # 👉 Gọi hàm
# summarize_labels_to_layers("E:\Data-Quan\TF2DeepFloorplan\TF2DeepFloorplan\Quantest\demo-2.dxf")

# def read_walls_from_dxf(file_path):
#     doc = ezdxf.readfile(file_path)
#     print("doc=",doc)
#     msp = doc.modelspace()
#     print("msp=",msp)

#     walls = []

#     for e in msp:
#         if e.dxftype() == 'LINE':
#             start = e.dxf.start
#             end = e.dxf.end
#             walls.append((start, end))
#         elif e.dxftype() == 'LWPOLYLINE':
#             points = e.get_points()
#             for i in range(len(points)-1):
#                 walls.append((points[i], points[i+1]))

#     return walls

# walls = read_walls_from_dxf("E:\Data-Quan\TF2DeepFloorplan\TF2DeepFloorplan\Quantest\demo-2.dxf")

# # Vẽ ra để xem thử
# for wall in walls:
#     x = [wall[0][0], wall[1][0]]
#     y = [wall[0][1], wall[1][1]]
#     plt.plot(x, y, 'k-')

# plt.axis('equal')
# plt.title("Walls from DXF")
# plt.show()
# def print_all_labels(dxf_file):
#     doc = ezdxf.readfile(dxf_file)
#     msp = doc.modelspace()

#     print("📋 Danh sách nhãn (TEXT / MTEXT):\n")

#     for entity in msp:
#         if entity.dxftype() == "TEXT":
#             text = entity.dxf.text
#             pos = entity.dxf.insert
#             print(f"[TEXT] '{text}' tại tọa độ {pos}")
        
#         elif entity.dxftype() == "MTEXT":
#             text = entity.text
#             pos = entity.dxf.insert
#             print(f"[MTEXT] '{text}' tại tọa độ {pos}")

# 👉 Gọi hàm với file DXF
# print_all_labels("E:\Data-Quan\TF2DeepFloorplan\TF2DeepFloorplan\Quantest\demo-2.dxf")

def display_all_entities_info(dxf_file):
    doc = ezdxf.readfile(dxf_file)
    msp = doc.modelspace()

    print(f"\n📄 Tên file: {dxf_file}")
    print(f"📏 Tổng số entity trong modelspace: {len(msp)}\n")

    for i, entity in enumerate(msp):
        print("entity.dxf=",entity.dxf)
        print(f"🔹 Entity #{i+1}")
        print(f"  - Type     : {entity.dxftype()}")
        print(f"  - Layer    : {entity.dxf.layer if entity.dxf.hasattr('layer') else 'N/A'}")
        print(f"  - Color    : {entity.dxf.color if entity.dxf.hasattr('color') else 'N/A'}")
        print(f"  - Linetype : {entity.dxf.linetype if entity.dxf.hasattr('linetype') else 'N/A'}")
        print(f"  - Lineweight: {entity.dxf.lineweight if entity.dxf.hasattr('lineweight') else 'N/A'}")
        
        # Tọa độ cho LINE
        if entity.dxftype() == "LINE":
            print(f"  - Start    : {entity.dxf.start}")
            print(f"  - End      : {entity.dxf.end}")

        # Tọa độ cho POLYLINE / LWPOLYLINE
        elif entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
            print("  - Vertices :")
            try:
                points = entity.get_points()
            except AttributeError:
                points = [v.dxf.location for v in entity.vertices()]
            for pt in points:
                print(f"    - {pt}")

        # TEXT / MTEXT
        elif entity.dxftype() in ["TEXT", "MTEXT"]:
            print(f"  - Text     : {entity.dxf.text if entity.dxf.hasattr('text') else entity.text}")
            print(f"  - Position : {entity.dxf.insert if entity.dxf.hasattr('insert') else 'N/A'}")

        # BLOCK hoặc INSERT
        elif entity.dxftype() == "INSERT":
            print(f"  - Block name: {entity.dxf.name}")
            print(f"  - Insert at : {entity.dxf.insert}")
            print(f"  - Scale     : {entity.dxf.xscale}, {entity.dxf.yscale}")
            print(f"  - Rotation  : {entity.dxf.rotation}")

        # ARC
        elif entity.dxftype() == "ARC":
            print(f"  - Center   : {entity.dxf.center}")
            print(f"  - Radius   : {entity.dxf.radius}")
            print(f"  - Start/End angle: {entity.dxf.start_angle}° → {entity.dxf.end_angle}°")

        # CIRCLE
        elif entity.dxftype() == "CIRCLE":
            print(f"  - Center   : {entity.dxf.center}")
            print(f"  - Radius   : {entity.dxf.radius}")
        
        print("-" * 40)

# 👉 Gọi hàm với file DXF của bạn
display_all_entities_info("E:\Data-Quan\TF2DeepFloorplan\TF2DeepFloorplan\Quantest\demo-2.dxf")

# import ezdxf
# import matplotlib.pyplot as plt

# # Đọc file DXF
# doc = ezdxf.readfile("E:\Data-Quan\TF2DeepFloorplan\TF2DeepFloorplan\Quantest\demo-2.dxf")
# msp = doc.modelspace()

# walls = []

# # Lấy các đối tượng LINE
# for e in msp:
#     # print("e",e)
#     if e.dxftype() == 'LINE':
#         start = e.dxf.start
#         end = e.dxf.end
#         walls.append(((start[0], start[1]), (end[0], end[1])))

# # Vẽ demo để kiểm tra
# for wall in walls:
#     x = [wall[0][0], wall[1][0]]
#     y = [wall[0][1], wall[1][1]]
#     plt.plot(x, y, 'k')  # màu đen
# plt.axis('equal')
# plt.title("Các đoạn tường từ DXF")
# plt.show()

# import ezdxf
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon

# # Mở file DXF
# file_path ="E:\Data-Quan\TF2DeepFloorplan\TF2DeepFloorplan\Quantest\demo-2.dxf"
# # Đọc file DXF
# doc = ezdxf.readfile(file_path)
# msp = doc.modelspace()

# lines = []       # chứa đoạn thẳng (tường)
# polygons = []    # chứa vùng kín từ polyline hoặc hatch

# for e in msp:
#     if e.dxftype() == 'LINE':
#         start = e.dxf.start
#         end = e.dxf.end
#         lines.append((start, end))

#     elif e.dxftype() == 'LWPOLYLINE':
#         if e.closed and len(e) >= 3:
#             points = [(p[0], p[1]) for p in e]
#             polygons.append(points)

#     elif e.dxftype() == 'HATCH':
#         for path in e.paths:
#             if path.path_type_flags & 1 and hasattr(path, "vertices"):
#                 points = [(v[0], v[1]) for v in path.vertices]
#                 if len(points) >= 3:
#                     polygons.append(points)

# # Vẽ hình
# fig, ax = plt.subplots(figsize=(10, 10))

# # Vẽ các đoạn thẳng (tường)
# for start, end in lines:
#     ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=0.8)

# # Vẽ các vùng hatch hoặc vùng kín
# for poly in polygons:
#     ax.add_patch(Polygon(poly, closed=True, facecolor='gray', edgecolor='black', alpha=0.4))

# ax.set_aspect('equal')
# ax.axis('off')
# plt.title("Tường và vùng tô từ DXF")
# plt.tight_layout()
# plt.show()
