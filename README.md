# D3GATTEN: Dense 3D Geometric Features Extraction Using Self-Attention

## Abstract
Detecting reliable geometric features is the key to creating a successful point-cloud registration. Point-cloud processing for extracting geometric features can be difficult, due to their invariance and the fact that most of them are corrupted by noise. In this work, we propose a new architecture, D3GATTEN, to solve this challenge, which allows to extract strong features, which later on can be used for point-cloud regis- tration, object reconstruction, and tracking. The key to our architecture is the use of the self-attention module to extract powerful features. Finally, compared with the most current methods, our architecture has achieved competitive results. Thoughtful tests were performed on the 3DMatch dataset, and it outperformed the existing state of the art. We  also demonstrated that getting the best features is the essence of point- cloud alignment.

## Overview
Our proposed network architecture

![net_archPNG](https://user-images.githubusercontent.com/22835687/169006113-ab8abe44-aee2-4cd3-ab24-d81cbad6e23c.PNG)

## Content
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Training & Evaluation](#training&evaluation)
- [Pretrained Model](#pretrainedmodel)
- [Results](#results)
- [Demo](#demo)
- [Smartcam demo - ADI](#smartcam-demo-adi)

## Prerequisites
1.  Create the environment and install the required libaries:
```
conda env create -f environment.yml
```
2. Compile the customized Tensorflow operators located in tf_custom_ops:
```
sh compile_op.sh
```
3. Compile the C++ extension module for python located in cpp_wrappers:
```
sh compile_wrappers.sh
```
## Data Preparation
The training set of 3DMatch can be downloaded from [here](https://mega.nz/file/fLBVXDqS#szY7USScX7T6wC0nZYNsnFDVJymcxECyzrRjFedrloU). It is generated by ```datasets/cal_overlap.py``` which select all the point cloud fragments pairs having more than 30% overlap.

## Training & Evaluation
1. The training on 3DMatch dataset can be done by running: 
```
python training_3DMatch.py
```
2. The testing can be done by running:
```
python test.py
```
## Pretrained Model
We provide the pre-trained model of 3DMatch in ```results/```.

## Results 
Example results on 3DMatch dataset. The point clouds that will be registered are represented by the first two columns (a) and (b). The standard deviation of the two point clouds is represented in the third column (c), and the result produced after performing the transformation is represented in the last column (d).

![reg_res](https://user-images.githubusercontent.com/22835687/169007947-fd57d63a-3737-4bb3-b8ee-8d3ca5be3bb4.PNG)


## Demo with own data
We provide a small demo to extract dense feature and detection score for two point cloud, and register them using RANSAC. To try the demo, please run:
```
python demo.py
```
![d3gatten](https://user-images.githubusercontent.com/22835687/169040368-dd1ad3b4-001c-49ec-97c3-f6d34eee06cc.gif)

## Smartcam demo - ADI
1. To use the camera, please follow the instructions [here](https://github.com/tamaslevente/trai/blob/master/README.md).
2. Getting images from the ADI smart camera:
```
rosrun pcl_ros pointcloud_to_pcd input:=/topic/name
```
3. Convert images to ```.ply``` format: 
```
pcl_pcd2ply [-format 0|1] [-use_camera 0|1] input.pcd output.ply
```
4. Run the demo
```
python demo.py --generate_features
```



