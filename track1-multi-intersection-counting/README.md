# AIC20 Track 1:Vehicle Counts by Class at Multiple Intersections
## Overview 

Our implementation comprised of: 

1/ Detected bounding box based on dla-centernet architecture

2/ Tracking car moving based on IoU overlapping

3/ Counting car based on densly sliding object through MOI region

<img src="pipeline_git.png" width=800 height=300 />

## Usage 

### Install Detector - CenterNet

#### Default GPU
I ran this code with **NVIDIA Tesla K80**. If you face a problem about GPU version, you can config GPU_version in CenterNet resources.

#### Download Dataset
1. Download images-set: [link](https://drive.google.com/open?id=1xFcfOEfAXjjzdrZbH3glOxv0rHIne8H7)

2. Download labels: [link](https://drive.google.com/file/d/1SWsjrSNaRp3CVe9h3Fu41ezkvXcGPy0_)

3. Create folders: 'abc', 'abc/images' and 'abc/labels' into 'data' folder.

4. Unzip images-set, then move all images from images-set folder to 'data/abc/images'.

5. Unzip labels, then move all json files to 'data/abc/labels'.

#### Install COCOAPI
```
cd AIC20/track1-multi-intersection-counting
COCOAPI = 'cocoapi'
git clone https://github.com/cocodataset/cocoapi.git 'cocoapi'
cd $COCOAPI/PythonAPI
make
python setup.py install --user
```

#### Install CenterNet
```
cd AIC20/track1-multi-intersection-counting
CenterNet_ROOT = 'CenterNet'
cd $CenterNet_ROOT
pip install -r requirements.txt
cd $CenterNet_ROOT/src/lib/external
python setup.py build_ext --inplace
```

#### Install DCN2
```
cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
python setup.py build develop
```

#### Pretrained Model - DLA
Download pretrained model: [ link ](
https://www.dropbox.com/s/q9jimptc5e8e2we/model_best_dla_1x.pth?dl=0)

#### Training
```
cd $CenterNet_ROOT/src
python main.py ctdet --exp_id abc_dla_34 --arch dla_34 --batch_size 32 --num_workers 4 --num_epochs 100
```
**Remember:** after training, all models are saving in 'exp/ctdet/<exp_id>/model_best.pth'

#### Inference
Firstly, you must have a pretrained model. Then:
```
cd $CenterNet_ROOT/src
python demo_video.py ctdet --arch dla_34 --load_model <link to model> --demo <link to video-folder>
```
**Remember:** after inference, your bounding box results are saving in 'CenterNet/Detection/bboxes_<video_name>'


### Reproduce tracking

```
cd AIC20

python test_iou.py
```

### Reproduce counting

```
cd AIC20/car_counter

python counter.py
```

### Acknowledgement

Source code for tracking car is built based on iou tracking of [High-Speed Tracking-by-Detection Without Using Image Information](https://github.com/bochinski/iou-tracker)



