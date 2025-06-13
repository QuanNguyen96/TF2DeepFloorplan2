đang sử dụng python 3.8.0
làm theo hướng dẫn trong file readme.txt
kiểm tra pip và python xem có cùng phiên bản ko (vì đang quản lý bằng pyenv, giống nvm của nodejs)
pip sẽ có thể nhận khác(theo python global của path trong system) do đó cần check lại nếu k thư viện tải về sẽ ăn theo python nằm trong pip

cài có thể lỗi pytype nếu chưa có c++ build trên window do đó phải cài  c++ build qua visual studio build 2022, sau đó tìm path file build.bat để chạy qua đây 
vd : "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
sau đó mới chạy : "pip install pytype"

-các bước chạy pygame
phải deploy lên trước 
vd python -m dfp.deploy --image ./30939153.jpg --weight log/store/G --postprocess --colorize --save ./output/output.jpg --loadmethod log







chạy server ứng dụng:
venv\Script\active ( vì là window)
-chạy server
uvicorn server:app --reload
-chạy các file python
python {path_file}


uvicorn server:app --reload
uvicorn server:app --reload
uvicorn server:app --reload