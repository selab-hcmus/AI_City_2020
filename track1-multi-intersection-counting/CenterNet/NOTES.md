# Hướng dẫn sử dụng CenterNet

## Các chức năng chính:
1. Train CenterNet
2. Train TinyDetector

## CenterNet
* Làm theo hướng dẫn trong ReadMe
* Đọc các note hỗ trợ trong Documents(JupyterNotebook for setup)

## Tiny Detector
* Sẽ note sau.

## Setup Dataset 
* Đọc noteCenterNet.docx
* Với dataset sử dụng trong AIC20 thì đó là abc.py nằm trong đường dẫn: "/home/xv/Documents/CenterNet2/src/lib/datasets/dataset/abc.py"
* Nếu data dạng voc thì có thể sử dụng CenterNet/src/tools/voc2coco/ để convert label (.xml) sang json format coco.

## Setup Inference
* Đã đi phần in kết quả đã inference ra màn hình (đọc note sẽ thấy)
* Đọc document/noteCenterNet.docx
* Có nhiều file demo.py được dùng cho các mục đích khác nhau:
1. demo.py: xuất output theo định dạng .pkl - default format của CenterNet.
2. demo_2.py: xuất output theo định dạng json, hãy đọc bên trong sẽ thấy cấu trúc file json. Lưu ý là bạn có thể chỉnh lại tùy thich.
3. demo_video.py: Inference chỉ áp dụng được cho thư mục chứa các file video và inference trên toàn bộ video đó. Chú ý: đã tắt đi phần tạo video kết quả. Muốn sử dụng lại hãy vào và xem code. (và xóa các #).