# PointNet
PointNet Implementation in pytorch.

![architecture](https://user-images.githubusercontent.com/37788686/156970649-79313266-4d0d-425e-a1ac-b1c40e0dd0e9.png)

## Project Structure

```
├── model.py              <- PointNet models implementation. <TODO>
│ 
├── dataloaders 
│   ├── modelnet.py         <- Dataloader of ModelNet40 dataset.
│   └── shapenet_partseg.py <- Dataloader of ShapeNet Part Annotation dataset. 
│
├── utils
│   ├── metrics.py          <- Easy-to-use code to compute metrics.
│   ├── misc.py             <- Point cloud normalization ft. and code to save rendered point clouds. 
│   └── model_checkpoint.py <- Automatically save model checkpoints during training.
│
├── train_cls.py          <- Run classification.
├── train_ae.py           <- Run auto-encoding
├── train_seg.py          <- Run part segmentation. <TODO>
├── visualization.ipynb   <- Simple point cloud visualization example code.
│
├── data                  <- Project data.
│   ├── modelnet40_ply_hdf5_2048     <- ModelNet40   
│   └── shapenet_part_seg_hdf5_data  <- ShapeNet Part Annotation
│
└── checkpoints           <- Directory storing checkpoints. 
    ├── classification
    │    └── mm-dd_HH-MM-SS/epoch=16-val_acc=88.6.ckpt
    ├── auto_encoding
    └── segmentation
```

## Classification
**_PA1: You will get the perfect score if you achieve test accuracy over 85%._**

You can train and test a model on ModelNet40 classification by running `train_cls.py`.
It will automatically download ModelNet40 dataset on `data` directory.
```
python train_cls.py [--epochs 100] [--batch_size 32] [--lr 1e-3] [--save]
```
If you pass `--save` option, only the topk model checkpoints will be saved in `checkpoints` directory. 

On ModelNet40 test set:

|                                | Overall Acc |
| ------------------------------ | ----------- |
| Paper                          | 89.2 %      |
| Ours (w/o feature trans.)      | 88.6 %      |
| Ours (w/ feature trans.)       | 87.7 %      | 

## Auto-Encoding
**_PA1: You will get the perfect score if you achieve chamfer distance lower than 0.005 on the test set._**

To train auto-encoder, run:
```
python train_ae.py
```
On ModelNet40 test set:
|        | Chamfer Dist. |
| ------ | ------------- |
| Ours   | 0.0043        |

## Part Segmentation
### TODOs
```
- model.py
- train_seg.py
```
**_PA2: You will get the perfect score if you achieve test mIoU over 80%._**

In `model.py`, you should implement PointNet part segmentation architecture on your own. We marked where you should fill the code in as `# TODO: Implement this`.

After implementing the network, call the network in `train_seg.py`. You should be able to see progress bars:

![image](https://user-images.githubusercontent.com/37788686/158202971-159e4dc3-199a-4cf2-9b12-c01059a06a4c.png)

ShapeNet part dataset will automatically be downloaded on `data` directory when `train_seg.py` is first executed.

We provide the code to measure instance mIoU in `utils/metrics.py`.

You can refer to other existing codes on the internet, but **DO NOT COPY** other students'. We will use a **plagiarism checker**.

On ShapeNet Part test set:
|        | ins. mIoU |
| ------ | --------- |
| Paper  | 83.7 %    |
| Ours   | 83.6 %    | 
## FYI
We provide some util codes to help you. Feel free to use them if you need.
- utils/model_checkpoint.py: tracks model's metric and automatically save only topk checkpoints.
- visualization.ipynb: simple visulization code on jupyter lab.

## Reference
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)

