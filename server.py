from fastapi import FastAPI,Request, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from TF2DeepFloorplan import predict_data,smooth_wall,cal_door_window

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ["*"] để cho tất cả
    allow_credentials=True,
    allow_methods=["*"],  # cho phép tất cả method như GET, POST,...
    allow_headers=["*"],  # cho phép tất cả header
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

# @app.get("/hello/{name}")
# def say_hello(name: str):
#     return {"message": f"Hello, {name}!"}


@app.post("/TF2-DeepFloorplan")
async def TF_DeepFloorplan(image: UploadFile = File(...)):  # field tên 'image' như frontend gửi):
  try:
    contents = await image.read()  # đọc dữ liệu file nhị phân
    filename = image.filename  # Lấy tên file upload
    # print("contents",image)
    
    # Tạo object file-like từ bytes
    image_stream = io.BytesIO(contents)
    
    # Mở ảnh với PIL
    pil_image = Image.open(image_stream)
    
    # print("pil_image",pil_image)
    results = predict_data(pil_image,filename)
    return JSONResponse(content={"data": results})
  except:
    return JSONResponse(content={"data": ""}, status_code=404)


@app.post("/smooth-wall")
async def Smooth_Wall(request: Request):  
  try:
    body = await request.json()
    if 'size' not in body or 'walls' not in body:
      return JSONResponse(content={"data": ""}, status_code=404)
    size=body["size"]
    walls=body["walls"]
    results = smooth_wall(size,walls)
    # print("ketqua=",results)
    return JSONResponse(content={"data": results})
  except:
    return JSONResponse(content={"data": ""}, status_code=404)


# @app.post("/get-file")
# async def getFile(image: UploadFile = File(...)):  # field tên 'image' như frontend gửi):
#   try:
#     contents = await image.read()  # đọc dữ liệu file nhị phân
#     print("contents",image)
    
#     # Tạo object file-like từ bytes
#     image_stream = io.BytesIO(contents)
    
#     # Mở ảnh với PIL
#     pil_image = Image.open(image_stream)
#     np_image = np.array(pil_image)
#     print("pil_image",pil_image)
#     print("np_image",np_image)
#     # results = predict_data()
#     return JSONResponse(content={"data": "ok",})
#   except:
#     return JSONResponse(content={"data": ""}, status_code=404)



@app.post("/cal-door-window")
async def calDoorWindow(request: Request):  
  try:
    body = await request.json()
    if 'points' not in body:
      return JSONResponse(content={"data": ""}, status_code=404)
    arr=body["points"]
    results = cal_door_window(arr)
    # print("ketqua=",results)
    return JSONResponse(content={"data": results})
  except:
    return JSONResponse(content={"data": ""}, status_code=404)
