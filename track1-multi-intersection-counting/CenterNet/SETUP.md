# Hướng dẫn cài đặt CenterNet

## Cài đặt môi trường
0. Môi trường thử nghiệm
CUDA >= 9.0; 
GPU: NVIDIA-Tesla K80

1. Cài đặt COCOAPI
```
COCOAPI = 'cocoapi'
git clone https://github.com/cocodataset/cocoapi.git 'cocoapi'
cd $COCOAPI/PythonAPI
make
python setup.py install --user
```

2. Cài đặt CenterNet
```
CenterNet_ROOT = 'CenterNet'
git clone https://github.com/xuan-vy-nguyen/CenterNet.git 'CenterNet'
cd $CenterNet_ROOT
pip install -r requirements.txt
cd $CenterNet_ROOT/src/lib/external
python setup.py build_ext --inplace
```

3. Cài đặt DCN2
```
cd $CenterNet_ROOT/src/lib/models/networks
rm -rf DCNv2
git clone https://github.com/CharlesShang/DCNv2
cd DCNv2
python setup.py build develop
```

## Dataset (có thể tham khảo file 'data/download_abc.sh')
1. Download dữ liệu ảnh tại link sau:
https://drive.google.com/open?id=1xFcfOEfAXjjzdrZbH3glOxv0rHIne8H7

2. Download labels tại link sau:
https://drive.google.com/file/d/1SWsjrSNaRp3CVe9h3Fu41ezkvXcGPy0_

3. Tạo thư mục 'abc', 'abc/images' và 'abc/labels' trong thư mục 'data'

4. Giải nén tất cả các file nén của dữ liệu ảnh, và copy tất cả các ảnh trong những thư mục trên vào 'data/abc/images' (Lưu ý, thư mục images chỉ lưu ảnh, không lưu thư mục nào cả).

5. Giải nén labels vừa tải về và copy 2 files nhãn cho tập train và val vào 'data/abc/labels' (tương tự trên, thư mục labels chỉ chứa file .json mà thôi)

## Bắt đầu train:
- Phần này đọc hướng dẫn của Tác giả CenterNet để hiểu rõ các tham số.
Lệnh mà mình dùng là: 
```
cd $CenterNet_ROOT/src
python main.py ctdet --exp_id abc_dla_34 --arch dla_34 --batch_size 32 --num_workers 4 --num_epochs 100
```
- Sau khi train xong thì model sẽ được lưu tại đường dẫn: 'exp/ctdet/<exp_id>/model_best.pth'


## Sử dụng Model đã train với tập dataset trên
Download pretrained-model tại link sau vào thư mục models với tên là "best_dla.pth"
https://www.dropbox.com/s/q9jimptc5e8e2we/model_best_dla_1x.pth?dl=0


## Inference
Sau khi đã train được model, dùng 'src/demo_video.py' để inference.
```
cd $CenterNet_ROOT/src
python demo_video.py ctdet --arch dla_34 --load_model <link đến model> --demo <link đến data cần infer>
```

