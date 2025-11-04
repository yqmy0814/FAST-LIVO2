# FAST-LIVO2
## 改进内容
1. 将sophus1.24.6和vikit整合至项目内，现在不再需要安装这两项依赖  
2. 添加了LRU内存管理，控制内存增长速度  
3. 使用谷歌风格对部分变量函数等进行了重命名，添加了部分注释  
4. 引入了glog和gflags（可按照后续步骤安装），在程序崩溃时可以便捷地找到出错的位置，便于调试，同时便于外部参数输入  
5. 添加了glibc malloc内存分配优化，避免长时间占用空闲内存(2025-09-03新增)  
6. 删去了LO模式和若干冗余变量，主要处理流程更加清晰(2025-09-03新增)  

## FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry

### 📢 News

- 🔓 **2025-01-23**: Code released!  
- 🎉 **2024-10-01**: Accepted by **T-RO '24**!  
- 🚀 **2024-07-02**: Conditionally accepted.

### 📬 Contact

For further inquiries or assistance, please contact [zhengcr@connect.hku.hk](mailto:zhengcr@connect.hku.hk).

## 1. Introduction

FAST-LIVO2 is an efficient and accurate LiDAR-inertial-visual fusion localization and mapping system, demonstrating significant potential for real-time 3D reconstruction and onboard robotic localization in severely degraded environments.

**Developer**: [Chunran Zheng 郑纯然](https://github.com/xuankuzcr)

<div align="center">
    <img src="pics/Framework.png" width = 100% >
</div>

### 1.1 Related video

Our accompanying video is now available on [**Bilibili**](https://www.bilibili.com/video/BV1Ezxge7EEi) and [**YouTube**](https://youtu.be/6dF2DzgbtlY).

### 1.2 Related paper

[FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry](https://arxiv.org/pdf/2408.14035)  

[FAST-LIVO2 on Resource-Constrained Platforms](https://arxiv.org/pdf/2501.13876)  

[FAST-LIVO: Fast and Tightly-coupled Sparse-Direct LiDAR-Inertial-Visual Odometry](https://arxiv.org/pdf/2203.00893)

[FAST-Calib: LiDAR-Camera Extrinsic Calibration in One Second](https://www.arxiv.org/pdf/2507.17210)

### 1.3 Our hard-synchronized equipment

We open-source our handheld device, including CAD files, synchronization scheme, STM32 source code, wiring instructions, and sensor ROS driver. Access these resources at this repository: [**LIV_handhold**](https://github.com/xuankuzcr/LIV_handhold).

### 1.4 Our associate dataset: FAST-LIVO2-Dataset
Our associate dataset [**FAST-LIVO2-Dataset**](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/zhengcr_connect_hku_hk/ErdFNQtjMxZOorYKDTtK4ugBkogXfq1OfDm90GECouuIQA?e=KngY9Z) used for evaluation is also available online.

### 1.5 Our LiDAR-camera calibration method
The [**FAST-Calib**](https://github.com/hku-mars/FAST-Calib) toolkit is recommended. Its output extrinsic parameters can be directly filled into the YAML file. 

## 2. Prerequisited

### 2.1 Ubuntu and ROS

Ubuntu 18.04~20.04.  [ROS Installation](http://wiki.ros.org/ROS/Installation).

### 2.2 PCL && Eigen && OpenCV

PCL>=1.8, Follow [PCL Installation](https://pointclouds.org/). 

Eigen>=3.3.4, Follow [Eigen Installation](https://eigen.tuxfamily.org/index.php?title=Main_Page).

OpenCV>=4.2, Follow [Opencv Installation](http://opencv.org/).

## 3. Build

Clone the repository and catkin_make:

```
cd ~/catkin_ws/src
git clone https://github.com/yqmy0814/FAST-LIVO2

# 安装glog和gflags,已安装则跳过
cd FAST-LIVO2/thirdparty
tar -xvf gflags-2.2.2.tar.gz 
cd gflags-2.2.2 
mkdir build && cd build 
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS=-fPIC .. 
make -j4 
sudo make install
cd ../..
tar -xvf glog-0.4.0.tar.gz 
cd glog-0.4.0 
mkdir build && cd build 
cmake -DBUILD_SHARED_LIBS=ON .. 
make -j4 
sudo make install

cd ~/catkin_ws
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## 4. Run our examples

Download our collected rosbag files via OneDrive ([**FAST-LIVO2-Dataset**](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/zhengcr_connect_hku_hk/ErdFNQtjMxZOorYKDTtK4ugBkogXfq1OfDm90GECouuIQA?e=KngY9Z)). 

```
roslaunch fast_livo mapping_avia.launch
rosbag play YOUR_DOWNLOADED.bag
```


## 5. License

The source code of this package is released under the [**GPLv2**](http://www.gnu.org/licenses/) license. For commercial use, please contact me at <zhengcr@connect.hku.hk> and Prof. Fu Zhang at <fuzhang@hku.hk> to discuss an alternative license.