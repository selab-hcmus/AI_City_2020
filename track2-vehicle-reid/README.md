# SELab - AICityChallenge 2020 - Track2: Vehicle Reidentification
------
This is our implementation for the solution submitted to Track2 [AICityChallenge 2020](https://www.aicitychallenge.org/). This track focuses on solving vehicle reidentification problem.
This year, three members from our group worked on different experiments with TripletNet, SiameseNet to tackle the mentioned challenge.
Repositories from other colleagues are mentioned below:
1. From `vltanh` (HCMUS-CNTN2016): [pytorch-reidentification](https://github.com/vltanh/pytorch-reidentification)
2. From `tqcuong98` (HCMUS-CNTN2016): [pytorch-multiview-reid](https://github.com/qcuong98/pytorch-multiview-reid)

## Installation
### Setup
A conda environment file info `environment.yml` is given, to install all requires packages:
```
$ conda env -f environment.yml
$ conda activate py3torch
```
Download Track 2 dataset, change the dataset folder directory under
```
configs/dataset_cfgs/aic20_vehicle_reid.yaml
```
Before starting every experiment, training and model configurations should be defined in
```
configs/model_cfgs
configs/train_cfgs
```
If you want to use your own model, loss, sampling objects, data augmentation strategy, etc. You can implement them inside the corrresponding folders inside `src/`. Remember to follow the interfaces and registered them under:
```
src/factories.py
```
### Training
Open a script file (e.g. `scripts/aic20_t2_trip_onlloss.sh`), change `--is_training` flag to `true` and execute the script:
```
$ ./scripts/aic20_t2_trip_onlloss.sh ${GPU_ID}
```
For your convenience, the default `GPU_ID` and `N_WORKERS` can be assigned under `scripts/master_env.sh` 
Results are given in the `logs/` folder, they includes the model (best score and checkpoints), log files for Tensorboard, and stdout. To use Tensorboard, under your log directory:
```
$ tensorboard --logdir=./ --port [YOUR PORT NUMBER]
```
**Note:** The best mAP we archived under this year challenge on our Easy Validation set at 90.29% is from the configurations under `scripts/aic20_t2_trip_onlloss.sh`.
### Testing
Similarly, change `--is_training` flag to `false`, specify the directory output under `--output` flag and the weight of your model under `--pretrained_model_path`.
#### Output
Execute the script again, and you can get the embeddings for your test and query set (`gal_emb.npy`, `que_emb.npy`), matrix distance between each pair of image from query and test set (`dist.npy`) and submission file (`track2.txt`).
#### Visualization
Thanks to `tmkhiem` (HCMUS-CNTN2016), we have this wonderful [web visualization tool](https://gitlab.com/Thevncore/aicitychallengevisualizer/-/tree/master) to view the prediction results.  

### Acknowledgement
The template for this repo was first initialized from the repo of `knmac`.
The implement of online triplet loss, batch sampling, mAP metrics computing are borrows from these two wonderful repos [Siamese-Triplet](https://github.com/adambielski/siamese-triplet) and [Triplet-Reid-Pytorch](https://github.com/CoinCheung/triplet-reid-pytorch)

Some of the loss functions are borrows from [Traffic-Brain-reid-aic19](https://github.com/he010103/Traffic-Brain/tree/master/AI-City-Vehicle-Reid)

For reranking, we used this repo [Re-ranking Person Re-identification with k-reciprocal Encoding[J]](https://github.com/zhunzhong07/person-re-ranking) from CVPR2017.