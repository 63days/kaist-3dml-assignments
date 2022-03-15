# Programming Assignments of CS492A: Machine Learning for 3D data
Professor: Minhyuk Sung (mhsung@kaist.ac.kr)  
TA: Juil Koo (63days@kaist.ac.kr)  
Last Updated: Mar 15, 2022.

## Introduction
PointNet is a fundamental yet powerful point cloud processing network.
In these programming assignments, we will learn about how PointNet works on various tasks in practice.
We will cover three tasks, _classification, auto-encoding and part segmentation_, step-by-step. 

There are two assignments as below. For each assignment, you need to submit both **a report (single .pdf file) and code (.zip file)** even when you just use the given code.

## PA1 (Classification and Auto-Encoding)
Due Date: Mar 21 (Mon)

For the first assignment, we provide all code for classification and auto-encoding on ModelNet dataset. You should just run the given code and report the results.
If you run `train_cls.py` and `train_ae.py`, you will be able to see the test results after training:

- `Classification`
<img width="647" alt="스크린샷 2022-03-14 오후 8 25 29" src="https://user-images.githubusercontent.com/37788686/158196370-c0239f51-1974-4934-bcfb-1582a0da61d4.png">

- `Auot-Encoder`
<img width="811" alt="스크린샷 2022-03-14 오후 8 24 20" src="https://user-images.githubusercontent.com/37788686/158196598-618ec2f8-bd47-4d3e-a24f-da7073b24631.png">

### What to hand in
__*Take screenshots like above and submit them on a single file so that we can see test results.*__

**_You will get the perfect score if results are on par with ours as below:_**

| Classification  | Auto-Encoding      |
| --------------  | -------------      |
| test acc >= 85% | test loss <= 0.005 |

If you implement chamfer distance yourself, take a mean across both batch and points.  
_i.e., given P1 [B,N,3] and P2 [B,N,3], Loss = ChamferDistance(P1, P2) / (B\*N), where B and N are batch size and num points respectively._

## PA2 (Part Segmentation)
Due Date: Mar 30 (Wed)

In this assignment, you will train and test PointNet with ShapeNet Part Annotation dataset.
We provide all base code, including a data loader, mIoU measurement and main loops code, except for a model implementation. 
You should implement `PointNetPartSeg` class in `model.py` by referring other models or existing code on the internet. 
You don't need to _completely follow the original implementation._   
As shown in Fig 9 in PointNet paper, the original implementation is somewhat complicated using T-Net and one hot vectors indicating the class of the input shape. 
You can skip these detailed components and _all you need to do is just to achieve test mIoU over 80%._ It would be achieved even with a way simpler architecture.  

After implementing the network, test the network by running `train_seg.py`. It will show the test mIoU and save some rendered outputs as `segmentation_samples.png`:

<img width="660" alt="스크린샷 2022-03-14 오후 10 37 41" src="https://user-images.githubusercontent.com/37788686/158196815-da63ec47-04b2-468a-9247-1dae80dc612e.png">
  
<img width="424" alt="스크린샷 2022-03-15 오전 12 03 56" src="https://user-images.githubusercontent.com/37788686/158200389-a2299163-8b60-4462-bc1a-491e87355b0f.png">

### What to hand in
__*Submit code as a zip file and report that includes the test mIoU result and rendered images.*__

**_You will get the perfect score if you achieve test mIoU over 80% and adequate qualitative results._**

