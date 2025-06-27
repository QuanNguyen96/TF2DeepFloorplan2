import ezdxf
from ezdxf.units import MM

# 1. Tạo file DXF mới
doc = ezdxf.new(setup=True)
doc.units = MM
msp = doc.modelspace()

# 2. Vẽ 1 phòng hình chữ nhật 5000 x 4000 mm (5m x 4m)
room_points = [(0, 0), (5000, 0), (5000, 4000), (0, 4000), (0, 0)]
msp.add_lwpolyline(room_points, close=True, dxfattribs={"layer": "ROOM"})

# 3. Thêm dòng text "Diện tích: 20.0 m2" vào giữa phòng
area_text = "Diện tích: 20.0 m²"
text = msp.add_text(area_text, dxfattribs={
    "height": 200,
    "layer": "ANNOTATION",
    "halign": 1,  # center
    "valign": 2   # middle
})
text.dxf.insert = (2500, 2000)

# 4. Tạo block ROOM_TAG chứa thuộc tính AREA
if "ROOM_TAG" not in doc.blocks:
    block = doc.blocks.new(name="ROOM_TAG")
    block.add_attdef(
        tag="AREA",
        text="20.0",
        insert=(0, 0),
        height=200
    )

# 5. Chèn block vào giữa phòng và gán giá trị thuộc tính
insert = msp.add_blockref("ROOM_TAG", (2500, 1800))
attrib = insert.add_attrib(tag="AREA", text="20.0", insert=(2500, 1800))
attrib.dxf.height = 200

# 6. Lưu file DXF
output_path = "sample_room_with_area.dxf"
doc.saveas(output_path)

print(f"✅ Đã tạo file DXF: {output_path}")
