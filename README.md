## KAIST CS492(A): Machine Learning for 3D Data (Spring 2022)<br>Programming Assignments
Instructor: [Minhyuk Sung](https://mhsung.github.io/) (mhsung@kaist.ac.kr)  
TA: Juil Koo (63days@kaist.ac.kr)  
Course webpage: [https://mhsung.github.io/kaist-cs492a-spring-2022/](https://mhsung.github.io/kaist-cs492a-spring-2022/)  
Last Updated: Mar 16, 2022.

## Introduction
[PointNet](https://arxiv.org/abs/1612.00593) is a fundamental yet powerful neural network processing point cloud data. In the following programming assignments, we will learn how to use PointNet for different tasks including _classification_, _auto-encoding_, and _segmentation_. 


## Programming Assignment 1 (Classification / Auto-Encoding)
<b>Due: Mar 21 (Mon) 23:59 KST</b>

The goal of the first assignment is to run our code for classification and auto-encoding and see how it works. **You just need to run our code and report the outputs!** Our code includes a script automatically downloading the [ModelNet](https://modelnet.cs.princeton.edu/) dataset and a dataloader. 

Run `train_cls.py` and `train_ae.py`, then you will see the test results after training:

- `Classification`
<img width="647" alt="스크린샷 2022-03-14 오후 8 25 29" src="https://user-images.githubusercontent.com/37788686/158196370-c0239f51-1974-4934-bcfb-1582a0da61d4.png">

- `Auto-Encoder`
<img width="811" alt="스크린샷 2022-03-14 오후 8 24 20" src="https://user-images.githubusercontent.com/37788686/158196598-618ec2f8-bd47-4d3e-a24f-da7073b24631.png">

### What to hand in
In a single document, write your name and student ID, add screenshots of the outputs (one for each task; thus two) like the ones above, and submit the document as a **PDF** file to KLMS.

If you want, you can implement the PointNet classification network and the autoencoder yourself! (e.g., using TensorFlow). In this case, submit your code (along with the snapshot document) as a **ZIP** file.

If you implement chamfer distance yourself, take the mean across both batches and points.  
_I.e., given P1 [B,N,3] and P2 [B,N,3], Loss = ChamferDistance(P1, P2) / (B\*N), where B and N are the batch size and the number of points, respectively._

### Grading
_You'll get a zero score if you implement yourself but don't submit your code._ Otherwise,

- 100% score:
  - If you submit on time, and
  - the document includes everything (your name, student ID, and two snapshots; one for classification and the other for auto-encoding),
  - and your results are on par with ours:

| Classification  | Auto-Encoding      |
| --------------  | -------------      |
| test acc >= 85% | test loss <= 0.005 |

- 50% score:
  - If you submit on time, but
    - either one of the snapshots is missing, OR
    - one of the results is worse than the threshold above.

- 0% score: Otherwise.


## Programming Assignment 2 (Segmentation)
<b>Due: Mar 30 (Wed) 23:59 KST</b>

The goal of the second assignment is to implement PointNet segmentation network based on our code.
Our code includes script automatically downloading the [ShapeNet](https://shapenet.org/) dataset, a dataloader, main training/test loops, and the evaluation code computing mIoU. **Implement the PointNet segmentation network in the `PointNetPartSeg` class of the `model.py` file. You can even use the code in the Internet to fill in this part!**

**Check out Figure 2 (not Figure 9) in the [paper](https://arxiv.org/abs/1612.00593)) for the details of the architecture, and feel free to leave out T-Nets**

Run `train_seg.py` after filling in the `PointNetPartSeg` class of the `model.py` file. Then, the code will show the test mIoU and save some images (`segmentation_samples.png`) showing the part segmentation results:

<img width="660" alt="스크린샷 2022-03-14 오후 10 37 41" src="https://user-images.githubusercontent.com/37788686/158196815-da63ec47-04b2-468a-9247-1dae80dc612e.png">
  
<img width="424" alt="스크린샷 2022-03-15 오전 12 03 56" src="https://user-images.githubusercontent.com/37788686/158200389-a2299163-8b60-4462-bc1a-491e87355b0f.png">

### What to hand in
Submit the followings to KLMS:
- A single **PDF** file including your name, student ID, a screenshot for mIoU, and the output image visualizing some segmentation results (like the ones above).
- A **ZIP** file including ALL the code (not just the part you implemented but all).

You can also implement the entire pipeline yourself if you want.

### Grading
_You'll get a zero score if you don't submit your code._ Otherwise,

- 100% score:
  - If you submit on time, and
  - the document includes everything (your name, student ID, and a mIoU snapshot, a segmentation visualization image),
  - and your result is on par with ours:

| Segmentation |
| -----------  |
| mIoU >= 80%  |

- 50% score:
  - If you submit on time, but
    - either one of the results (a mIoU snapshot or a segmentation visualization image) is missing, OR
    - the result is worse than the threshold above.

- 0% score: Otherwise.
