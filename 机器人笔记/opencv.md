- [opencvgui特性](#opencvgui特性)
  - [2\_1\_图像入门](#2_1_图像入门)
    - [读取图像](#读取图像)
    - [创建和显示窗口](#创建和显示窗口)
    - [显示图像](#显示图像)
    - [写入图像](#写入图像)
    - [键盘输入](#键盘输入)
    - [matplotlib](#matplotlib)
    - [练习题](#练习题)
  - [2\_2\_视频入门](#2_2_视频入门)
    - [从相机读取视频](#从相机读取视频)
    - [从文件播放视频](#从文件播放视频)
    - [保存视频](#保存视频)
  - [2\_3\_OpenCV中的绘图功能](#2_3_opencv中的绘图功能)
    - [练习题](#练习题-1)
  - [2\_4\_鼠标作为画笔](#2_4_鼠标作为画笔)
    - [练习题](#练习题-2)
  - [2\_5\_轨迹栏作为调色板](#2_5_轨迹栏作为调色板)
    - [练习](#练习)
- [核心操作](#核心操作)
  - [3\_1\_图像的基本操作](#3_1_图像的基本操作)
    - [添加边框](#添加边框)
  - [3\_2\_图像上的算法运算](#3_2_图像上的算法运算)
    - [加法](#加法)
    - [图像融合](#图像融合)
    - [按位运算](#按位运算)
    - [练习题](#练习题-3)
  - [3\_3\_性能衡量和提升技术](#3_3_性能衡量和提升技术)
    - [获取执行时间](#获取执行时间)
    - [openc优化](#openc优化)
    - [IPython中衡量性能](#ipython中衡量性能)
    - [性能优化技术](#性能优化技术)
- [OpenCV中的图像处理](#opencv中的图像处理)
  - [4\_1\_改变颜色空间](#4_1_改变颜色空间)
    - [练习题](#练习题-4)
  - [4\_2\_图像几何变换](#4_2_图像几何变换)
    - [缩放](#缩放)
    - [平移](#平移)
    - [旋转](#旋转)
    - [仿射变换](#仿射变换)
    - [透视变换](#透视变换)
  - [4\_3\_图像阈值](#4_3_图像阈值)
    - [简单阈值](#简单阈值)
    - [自适应阈值](#自适应阈值)
    - [Otsu的二值化](#otsu的二值化)
  - [4\_4\_图像平滑](#4_4_图像平滑)
    - [2D卷积（图像过滤）](#2d卷积图像过滤)
    - [图像模糊](#图像模糊)
      - [平均](#平均)
      - [高斯模糊](#高斯模糊)
      - [中位模糊](#中位模糊)
      - [双边滤波](#双边滤波)
  - [4\_5\_形态转换](#4_5_形态转换)
    - [腐蚀](#腐蚀)
    - [膨胀](#膨胀)
    - [开运算](#开运算)
    - [闭运算](#闭运算)
    - [形态学梯度](#形态学梯度)
    - [顶帽](#顶帽)
    - [黑帽](#黑帽)
    - [创建所需内核](#创建所需内核)
  - [4\_6\_图像梯度](#4_6_图像梯度)
    - [Sobel 和 Scharr 算子](#sobel-和-scharr-算子)
      - [sobel算子](#sobel算子)
    - [Scharr算子](#scharr算子)
    - [Laplacian 算子](#laplacian-算子)
  - [4\_7\_Canny边缘检测](#4_7_canny边缘检测)
  - [4\_8\_图像金字塔](#4_8_图像金字塔)
    - [高斯金字塔\\拉普拉斯金字塔](#高斯金字塔拉普拉斯金字塔)
  - [4\_9\_1\_OpenCV中的轮廓](#4_9_1_opencv中的轮廓)
    - [绘制轮廓](#绘制轮廓)
  - [4\_9\_2\_轮廓特征](#4_9_2_轮廓特征)
    - [轮廓面积](#轮廓面积)
    - [轮廓周长](#轮廓周长)
    - [轮廓近似](#轮廓近似)
    - [轮廓凸包](#轮廓凸包)
    - [检查凸度](#检查凸度)
    - [边界矩阵](#边界矩阵)
    - [最小闭合圈](#最小闭合圈)
    - [拟合一个椭圆](#拟合一个椭圆)
    - [拟合直线](#拟合直线)
  - [4\_9\_3\_轮廓属性](#4_9_3_轮廓属性)
    - [长宽比 aspect\_ratio = float(w)/h](#长宽比-aspect_ratio--floatwh)
    - [范围](#范围)
    - [坚实度](#坚实度)
    - [等效直径](#等效直径)
    - [取向](#取向)
    - [掩码和像素点](#掩码和像素点)
    - [最大值，最小值和它们的位置](#最大值最小值和它们的位置)
    - [平均颜色或平均强度](#平均颜色或平均强度)
    - [极端点](#极端点)
  - [4\_9\_4\_轮廓：更多属性](#4_9_4_轮廓更多属性)
    - [凸性缺陷](#凸性缺陷)
    - [点多边形测试](#点多边形测试)
    - [形状匹配](#形状匹配)
  - [4\_9\_5\_轮廓分层](#4_9_5_轮廓分层)
    - [RETR\_LIST](#retr_list)
    - [RETR\_EXTERNAL](#retr_external)
    - [RETR\_CCOMP](#retr_ccomp)
    - [RETR\_TREE](#retr_tree)
  - [4\_10\_1\_直方图-1：查找，绘制，分析](#4_10_1_直方图-1查找绘制分析)
    - [绘制直方图](#绘制直方图)
  - [4\_10\_2\_直方图-2：直方图均衡](#4_10_2_直方图-2直方图均衡)
    - [CLAHE（对比度受限的自适应直方图均衡）](#clahe对比度受限的自适应直方图均衡)
  - [4\_10\_3\_直方图3：二维直方图](#4_10_3_直方图3二维直方图)
    - [绘制二维直方图](#绘制二维直方图)
  - [4\_10\_4\_直方图-4：直方图反投影](#4_10_4_直方图-4直方图反投影)
  - [4\_11\_傅里叶变换](#4_11_傅里叶变换)
    - [numpy 中](#numpy-中)
    - [OpenCV中的傅里叶变换](#opencv中的傅里叶变换)
    - [性能优化](#性能优化)
  - [4\_12\_模板匹配](#4_12_模板匹配)
  - [4\_13\_霍夫线变换](#4_13_霍夫线变换)
    - [概率霍夫变换](#概率霍夫变换)
  - [4\_14\_霍夫圈变换](#4_14_霍夫圈变换)
  - [4\_15\_图像分割与分水岭算法](#4_15_图像分割与分水岭算法)
  - [4\_16\_交互式前景提取使用GrabCut算法](#4_16_交互式前景提取使用grabcut算法)
- [特征检测与描述](#特征检测与描述)
  - [5\_1\_理解特征](#5_1_理解特征)
  - [5\_2\_哈里斯角检测](#5_2_哈里斯角检测)
    - [SubPixel精度的转角cv.cornerSubPix](#subpixel精度的转角cvcornersubpix)
  - [5\_3\_Shi-Tomasi拐角探测器和良好的跟踪功能](#5_3_shi-tomasi拐角探测器和良好的跟踪功能)
  - [5\_4\_SIFT（尺度不变特征变换）简介](#5_4_sift尺度不变特征变换简介)
    - [计算描述符](#计算描述符)
  - [5\_5\_SURF简介（加速的强大功能）](#5_5_surf简介加速的强大功能)
  - [5\_6\_用于角点检测的FAST算法](#5_6_用于角点检测的fast算法)
  - [5\_7\_BRIEF（二进制的鲁棒独立基本特征）](#5_7_brief二进制的鲁棒独立基本特征)
  - [5\_8\_ORB（定向快速和旋转简要）](#5_8_orb定向快速和旋转简要)
  - [5\_9\_特征匹配](#5_9_特征匹配)
    - [Brute-Force匹配器的基础](#brute-force匹配器的基础)
    - [基于匹配器的FLANN](#基于匹配器的flann)
  - [5\_10\_特征匹配+单应性查找对象](#5_10_特征匹配单应性查找对象)
- [视频分析](#视频分析)
  - [6\_1\_如何使用背景分离方法](#6_1_如何使用背景分离方法)
  - [6\_2\_Meanshift和Camshift](#6_2_meanshift和camshift)
    - [Meanshift](#meanshift)
    - [Camshift](#camshift)
  - [6\_3\_光流](#6_3_光流)
    - [Lucas-Kanade](#lucas-kanade)
    - [Gunner Farneback的算法](#gunner-farneback的算法)
- [相机校准和3D重建](#相机校准和3d重建)
  - [7\_1\_相机校准](#7_1_相机校准)
    - [拍摄图像并对其进行扭曲。](#拍摄图像并对其进行扭曲)
    - [重投影误差](#重投影误差)
  - [7\_2\_姿态估计](#7_2_姿态估计)
  - [7\_3\_对极几何](#7_3_对极几何)
  - [7\_4\_立体图像的深度图](#7_4_立体图像的深度图)
- [机器学习](#机器学习)
  - [8\_1\_理解KNN](#8_1_理解knn)
  - [8\_2\_使用OCR手写数据集运行KNN](#8_2_使用ocr手写数据集运行knn)
  - [8\_3\_理解SVM](#8_3_理解svm)
  - [8\_4\_使用OCR手写数据集运行SVM](#8_4_使用ocr手写数据集运行svm)
  - [8\_5\_理解K均值聚类](#8_5_理解k均值聚类)
  - [8\_6\_OpenCV中的K均值](#8_6_opencv中的k均值)
- [计算摄影学](#计算摄影学)
  - [9\_1\_图像去噪](#9_1_图像去噪)
  - [9\_2\_图像修补](#9_2_图像修补)
  - [9\_3\_高动态范围](#9_3_高动态范围)
- [目标检测](#目标检测)
  - [10\_1\_级联分类器](#10_1_级联分类器)
  - [10\_2\_级联分类器训练](#10_2_级联分类器训练)
# opencvgui特性
## 2_1_图像入门
### 读取图像
```
cv.imread(img,FLAG)
```
1. cv.IMREAD_COLOR（1）： 默认，加载彩色图片忽视透明度
2. cv.IMREAD_GRAYSCALE（0）：灰度图
3. cv.IMREAD_UNCHANGED（-1）：加载图像，包括alpha通道
### 创建和显示窗口
```
cv.namedWindow(windowname，cv.WINDOW_NORMAL)
cv.imshow('image'，img)
cv.destroyWindow(windowname)
cv.destroyAllWindows()
```
### 显示图像
```
cv.imshow(windowname,img)
```
### 写入图像
```
cv.imwrite(path,img)
```
### 键盘输入
```
cv.waitKey(t)
```
t时间内等待输入，如果t=0则一直等待
### matplotlib
```
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('messi5.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # 隐藏 x 轴和 y 轴上的刻度值
plt.show()
```
### 练习题
尝试在OpenCV中加载彩色图像并将其显示在Matplotlib中
```
path = os.path.join('images','02.jpg')
img = cv2.imread(path,1)
# b,g,r = cv2.split(img)
# rgb = cv2.merge([r,g,b])
rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(rgb,cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) 
plt.show()
```
## 2_2_视频入门
### 从相机读取视频
```
cap = cv.VideoCapture(0) #捕获0号摄像头
if not cap.isOpened(): #判断是否成功打开
    print("Cannot open camera")
    exit()
while True:
 # 逐帧捕获
    ret, frame = cap.read() 
 # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.waitKey(1)
cap.release() #释放
```
```
cap.get(cv.CAP_PROP_FRAME_WIDTH)
cap.get(cv.CAP_PROP_FRAME_HEIGHT) #查看宽高，默认640x480
#修改宽高
ret = cap.set(cv.CAP_PROP_FRAME_WIDTH,320)
ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT,240)
```
### 从文件播放视频
```
#主体改为这个就行
cap = cv.VideoCapture('vtest.avi')
while cap.isOpened():
    ...
```
### 保存视频
```
# 定义编解码器并创建VideoWriter对象
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
while cap.isOpened():
    ret, frame = cap.read()
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
```
## 2_3_OpenCV中的绘图功能
cv2.LINE_AA抗锯齿的线条
linewidth正数表示线宽，否则表示填充
```
#画线
cv2.line(img,start_pos,end_pos,color,linewidth，lineType)
#画矩阵
cv2.rectangle(img,left_top_pos,right_bottom_pos,color,linewidth)
#画圆圈
cv2.circle(img,center,r,color,linewidth)
#画椭圆
cv2.ellipse(img,center,axes,angle,startAngle,endAngle,color,thickness = 1,lineType = LINE_8,shift = 0)	
axes:(长轴长度短轴长度)
angle椭圆沿逆时针旋转的角度
startAngle,endAngle 分别为0，360时给出完整的圆
#画多边形
cv2.polylines(img,[pts],True,(0,255,255))
pts:点数*1*2存储顶点坐标
#添加文本
cv.putText(img,txt,pos, font,font_size,color,linewidth,linetype)
```
### 练习题
```
import cv2
import numpy as np
import os
from IPython.display import Image, display
logo = cv2.imread(os.path.join('images','logo.png'))
cv2.imwrite('temp.jpg',logo)
display(Image('temp.jpg'))
h,w,c = img.shape
print(h,w,c)
img = np.zeros_like(logo)
color_red = (68,42,255)
color_green = (103,218,139)
color_blue = (255,141,18)
color_white = (255,255,255)
color_black = (0,0,0)
max_r = 50
min_r = 20
center_red = (100,50)
center_green = (49,140)
center_blue = (153,140)
cv2.circle(img,center_red,max_r,color_red,-1)
cv2.circle(img,center_red,min_r,color_black,-1)
cv2.ellipse(img, center_red, (max_r, max_r + 2), 0, 60, 120, color_black, -1,cv2.LINE_AA)
cv2.circle(img,center_green,max_r,color_green,-1)
cv2.circle(img,center_green,min_r,color_black,-1)
cv2.ellipse(img, center_green, (max_r, max_r + 2), 240,60, 120, color_black, -1,cv2.LINE_AA)
cv2.circle(img,center_blue,max_r,color_blue,-1)
cv2.circle(img,center_blue,min_r,color_black,-1)
cv2.ellipse(img, center_blue, (max_r, max_r + 2), 180, 60, 120, color_black, -1,cv2.LINE_AA)
cv2.putText(img,'OpenCV',(0, h - 15),cv2.FONT_HERSHEY_DUPLEX, 1.5, color_white,3)
cv2.imwrite('temp.jpg',img)
display(Image('temp.jpg'))
```
![img](logo.png)
![img](temp.jpg)
## 2_4_鼠标作为画笔
```
def draw_circle(event,x,y,flags,param)
```
### 练习题
```
import numpy as np
import cv2 as cv
drawing = False # 如果按下鼠标，则为真
mode = True # 如果为真，绘制矩形。按 m 键可以切换到曲线
ix,iy = -1,-1
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
        else:
            cv.circle(img,(x,y),5,(0,0,255),1)
img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
while(1):
    cv.imshow('image',img)
    if cv.waitKey(1) & 0xFF == 27:
        break
cv.destroyAllWindows()
```
## 2_5_轨迹栏作为调色板
```
#创建轨迹栏
cv2.createTrackbar(trackerbar_name,windowname,min,max,回调函数)
#创建开关
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)
#读取轨迹位置
cv2.getTrackbarPos(trackerbar_name,windowname)
```
### 练习
```
import numpy as np
import cv2 as cv
def nothing(x):
    pass
flag = False
def draw_circle(event,x,y,flags,param):
    global flag
    r = cv.getTrackbarPos('R','image')
    g = cv.getTrackbarPos('G','image')
    b = cv.getTrackbarPos('B','image')
    lw = cv.getTrackbarPos('linewidth','image')
    if event  == cv.EVENT_MOUSEMOVE and flag:
        cv.circle(img,(x,y),lw,(b,g,r),-1)
    elif event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),lw,(b,g,r),-1)
        flag = True
    elif event == cv.EVENT_LBUTTONUP:
        cv.circle(img,(x,y),lw,(b,g,r),-1)
        flag = False
img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)
cv.createTrackbar('linewidth','image',1,50,nothing)
while(1):
    cv.imshow('image',img)
    if cv.waitKey(1) & 0xFF == 27:
        break
cv.destroyAllWindows()
```
# 核心操作
## 3_1_图像的基本操作
```
#索引方式访问更改
img[100,100]
img[100,100,0]
img[280:340, 330:390]
#对于单个像素访问，Numpy数组方法array.item()和array.itemset()被认为更好
img.item(10,10,2)
img.itemset((10,10,2),100)
#图像形状
img.shape
#像素总数
img.size
#图像数据类型
img.dtype
#拆分合并通道数
b,g,r = cv.split(img)
img = cv.merge((b,g,r))
#cv.split()比索引更加耗时
b = img [:, :, 0]
img [:, :, 0] = b
```
### 添加边框
```
#top，bottom，left，right边界各方向宽度
cv.copyMakeBorder(img,top，bottom，left，right,borderType)
```
1. cv.BORDER_CONSTANT 恒定彩色边框
2. 2.cv.BORDER_REFLECT 边框将是边框元素的镜像
## 3_2_图像上的算法运算
### 加法
```
#OpenCV加法是饱和运算，而Numpy加法是模运算
#OpenCV功能将提供更好的结果
cv.add(x,y) 或 x + y
```
### 图像融合
$$G(x)= (1 - \alpha)f_0(x)+ \alpha f_1(x)$$
$$cv.addWeighted(img1,\alpha,img2,\beta,\gamma)应用如下公式$$
$$dst=\alpha \cdot img1+\beta \cdot img2 + \gamma$$
### 按位运算
cv.bitwise_and(img1,img2,mask) / or / xor / not

### 练习题
```
def main():
    path = 'aadata'
    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    pic_list = os.listdir(path)
    for i in range(1,len(pic_list)):
        img1 = cv2.imread(os.path.join(path,pic_list[i-1]))
        img2 = cv2.imread(os.path.join(path,pic_list[i]))
        for j in range(101):
            a = 1 -  0.01 * j
            img = cv2.addWeighted(img1,a,img2,1-a,0)
            cv2.imshow('img',img)
            if cv2.waitKey(1) & 0xFF == 27:
                return 
    cv2.destroyAllWindows()
```
## 3_3_性能衡量和提升技术
### 获取执行时间
cv.getTickCount
返回从参考事件到调用此函数那一刻之间的时钟周期数
```
#获取时钟频率
cv.getTickFrequency
#获取执行时间
e1 = cv.getTickCount()
# 你的执行代码
e2 = cv.getTickCount()
time = (e2 - e1)/ cv.getTickFrequency()
#用time模块(如time.time)可得到同样效果
```
### openc优化
```
cv2.cvUseoptimized() #检查是否启用优化
cv2.cvSetuseoptimized(bool) 启用(禁用)优化
```
### IPython中衡量性能
```
%测时 y=x**2
#OpenCV函数比Numpy函数要快,可能会有例外，尤其是当Numpy处理视图而不是副本时
```
### 性能优化技术
[Python 优化](http://wiki.python.org/moin/PythonSpeed/PerformanceTips)

[Scipy讲义- 高级Numpy](http://scipy-lectures.github.io/advanced/advanced_numpy/index.html#advanced-numpy)

[IPython 中时序和性能优化](http://pynash.org/2013/03/06/timing-and-profiling/)
# OpenCV中的图像处理
## 4_1_改变颜色空间
```
cvtColor(input_image, flag)
```
|flag|
|---|
|cv.COLOR_BGR2GRAY|
|cv.COLOR_BGR2HSV|

HSV颜色空间
1. H:[0,179]  色相
2. S:[0,255]  饱和度
3. V:[0,255]  明度
### 练习题
```
    img = cv2.imread(os.path.join(path,'pics (1).jpg'))
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    range_blue = np.array([[90,100,100],[179,255,255]])
    range_green = np.array([[60,100,100],[80,255,255]])
    range_red = np.array([[0,100,100],[50,255,255]])

    mask_blue = cv2.inRange(hsv,range_blue[0],range_blue[1])
    mask_green = cv2.inRange(hsv,range_green[0],range_green[1])
    mask_red = cv2.inRange(hsv,range_red[0],range_red[1])

    blue = cv2.bitwise_and(img,img,mask=mask_blue)
    green = cv2.bitwise_and(img,img,mask=mask_green)
    red = cv2.bitwise_and(img,img,mask=mask_red)
    
    two = cv2.bitwise_or(red,green,mask = None)
    three = cv2.bitwise_or(two,blue,mask=None)
    cv2.imshow('img',three)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```
## 4_2_图像几何变换
### 缩放
```
resize(src,dsize,fx=0,fy=0,interpolation)
interpolation(插值方法)
res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
#或者
height, width = img.shape[:2]
res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
```
|interpolation|插值方法|
|---|---|
|INTER_NEAREST|最邻近插值|
|INTER_LINEAR|双线性插值|
|INTER_AREA||
|INTER_CUBIC|4x4像素邻域内的双立方插值|
|INTER_LANCZOS4|8x8像素邻域内的Lanczos插值|
### 平移
M 仿射变换矩阵，2行3列
$$M = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix}$$
$$\begin{bmatrix}x^{'} \\ y^{'}\end{bmatrix}M = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix}\begin{bmatrix}x \\ y \\1 \end{bmatrix}$$
```
dsize:(width,height)输出图像大小
cv2.warpAffine(src, M, dsize)
```
### 旋转
正常旋转的变换矩阵
$$M = \begin{bmatrix} cos\theta & -sin\theta \\ sin\theta & cos\theta \end{bmatrix}$$
opencv的仿射变换矩阵,可以选择中心和缩放
$$\begin{bmatrix} \alpha & \beta & (1- \alpha ) \cdot center.x - \beta \cdot center.y \\ - \beta &
\alpha & \beta \cdot center.x + (1- \alpha ) \cdot center.y \end{bmatrix}$$
$$\begin{array}{l} \alpha = scale \cdot \cos \theta , \\ \beta = scale \cdot \sin \theta \end{array}$$
```
#逆时针90度
# cols-1 和 rows-1 是坐标限制
rows,cols = img.shape
#cv.getRotationMatrix2D(contor,angle,scale)
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(cols,rows))
```
### 仿射变换
```
pts1:输入图像的三个点的坐标 3x2
pts2:输出图像的三个点的对应坐标 3x2
M = cv.getAffineTransform(pts1,pts2)
cv.warpAffine(img,M,(cols,rows))
```
### 透视变换
3x3变换矩阵
```
#pts1 输入图像的四个点，不能有三个点在一条线
M = cv.getPerspectiveTransform(pts1,pts2)
cv.warpPerspective(img,M,(cols,rows))
```
## 4_3_图像阈值
### 简单阈值
像素值小于阈值，则将其设置为0，否则将其设置为最大值
```
gray:输入灰度图
ret:返回使用的阈值
thresh1:返回阈值后的图像
ret,thresh1 = cv.threshold(gray,阈值,max_val,cv.THRESH_BINARY)
```
cv2.THRESH_BINARY|大于阈值赋值maxVal否则0
|--|--|
cv2.THRESH_BINARY_INV|小于阈值赋值maxVal否则0
cv2.THRESH_TRUNC|大于阈值赋值maxval否则不变
cv2.THRESH_TOZERO|大于阈值不变，否则设为0
cv2.THRESH_TOZERO_INV|大于阈值赋值0，否则不变
### 自适应阈值
```
cv.adaptiveThreshold(src,maxValue,
adaptiveMethod,thresholdType,blockSize,C)
src：灰度化的图片
maxValue：满足条件的像素点需要设置的灰度值
adaptiveMethod：自适应方法。有2种：ADAPTIVE_THRESH_MEAN_C 或 ADAPTIVE_THRESH_GAUSSIAN_C
thresholdType：二值化方法，可以设置为THRESH_BINARY或者THRESH_BINARY_INV
blockSize：分割计算的区域大小，取奇数
C：常数，每个区域计算出的阈值的基础上在减去这个常数作为这个区域的最终阈值，可以为负数
```
- adaptiveMethod
  - cv.ADAPTIVE_THRESH_MEAN_C::阈值是邻近区域的平均值减去常数C
  - cv.ADAPTIVE_THRESH_GAUSSIAN_C:阈值是邻域值的高斯加权总和减去常数。
### Otsu的二值化
Otsu的方法从图像直方图中确定最佳全局阈值
```
cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
```
- Otsu的二值化有一些优化。您可以搜索并实现它。
## 4_4_图像平滑
- 低通滤波器（LPF），高通滤波器（HPF）
- LPF有助于消除噪声，使图像模糊等。HPF滤波器有助于在图像中找到边缘。
### 2D卷积（图像过滤）
```
#将内核与图像进行卷积
ddepth指定输出数据类型，通常设为-1与输入相同
cv.filter2D(img,ddepth,kernel)
```
### 图像模糊
- 通过将图像与低通滤波器内核进行卷积来实现图像模糊。这对于消除噪音很有用。它实际上从图像中消除了高频部分（例如噪声，边缘）。
#### 平均
- 获取内核区域下所有像素的平均值，并替换中心元素
```
cv.blur(img,kernel_size)
#normalize为1时表示进行归一化，此时同均值滤波否则为领域像素之和
cv2.boxFilter（img,-1,kernel_size,normalize）
```
#### 高斯模糊
- 高斯模糊对于从图像中去除高斯噪声非常有效。
```
blur = cv.GaussianBlur(img,ksize,sigmax,sigmay)
```
#### 中位模糊
- 用邻域内所有像素值的中间值来替代当前像素点的像素值
```
cv2.medianBlur（img,ksize）
```
#### 双边滤波
- 在去除噪声的同时保持边缘清晰锐利非常有效，该操作速度较慢
```
#d:过滤过程中每个像素邻域的直径
#sigmaColor，颜色空间滤波器的sigma值。这个参数的值越大，就表明该像素邻域内有更宽广的颜色会被混合到一起，产生较大的半相等颜色区域。
#sigmaSpace坐标空间中滤波器的sigma值，坐标空间的标注方差。他的数值越大，意味着越远的像素会相互影响，从而使更大的区域足够相似的颜色获取相同的颜色。当d>0，d指定了邻域大小且与sigmaSpace无关。否则，d正比于sigmaSpace。
cv.bilateralFilter(img,d,sigmacolor,sigmaspace)
```
## 4_5_形态转换
- 形态变换是一些基于图像形状的简单操作。通常在二进制图像上执行
### 腐蚀
- 有助于去除小的白色噪声，分离两个连接的对象
- 内核滑动通过图像(在2D卷积中)。原始图像中的一个像素(无论是1还是0)只有当内核下的所有像素都是1时才被认为是1，否则它就会被侵蚀(变成0)。
```
kernel = np.ones((5,5),np.uint8)
cv.erode(img,kernel,iterations = 1)
```
### 膨胀
- 如果内核下的至少一个像素为“ 1”，则像素元素为“ 1”。
- 通常，在消除噪音的情况下，腐蚀后会膨胀
- 在连接对象的损坏部分时也很有用
```
cv.dilate(img,kernel,iterations = 1)
```
### 开运算
- 侵蚀然后扩张
```
cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
```
### 闭运算
- 先扩张然后再侵蚀
- 在关闭前景对象内部的小孔或对象上的小黑点时很有
用
```
cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
```
### 形态学梯度
- 梯度=膨胀-腐蚀:膨胀和腐蚀图像的差，可以提取边缘
```
cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
```
### 顶帽
- 输入图像和图像开运算之差
```
#开运算可以消除暗背景下的高亮区域，那么如果用原图减去开运算结果就可以得到原图中灰度较亮的区域所以又称白顶帽变换
cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
```
### 黑帽
- 输入图像和图像闭运算之差
```
#闭运算可以删除亮背景下的暗区域，那么用原图减去闭运算结果就可以得到原图像中灰度较暗的区域，所以又称黑底帽变换。
cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
```
### 创建所需内核
```
#矩形
cv.getStructuringElement(cv.MORPH_RECT,(5,5))
#椭圆内核
cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
#十字内核
cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
```
## 4_6_图像梯度
- OpenCV提供三种类型的梯度滤波器或高通滤波器，即Sobel，Scharr和Laplacian
### Sobel 和 Scharr 算子
#### sobel算子
x反方向梯度算子
$$\begin{matrix}-1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{matrix} \rightarrow G_{x}$$
y方向梯度算子
$$\begin{matrix}-1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{matrix}\rightarrow G_{y}$$
$$G=\sqrt{G_{x}^{2}+G_{y}^{2}}或G=\lvert G_{x}\lvert+\lvert G_{y}\lvert$$
```
#dx,dy仅计算x方向则dx=1，dy=0，通常分别取两方向再做按权相加，直接dx=1，dy=1效果不好
cv2.Sobel(img,-1,dx,dy,ksize)
#可以用安权值加法合并x，y方向梯度
cv2.addWeight(img1,w1,img2,w2)
#取绝对值函数
cv2.convertScaleAbs(img)
```
### Scharr算子
x反方向梯度算子
$$\begin{matrix}-3 & 0 & 3 \\ -10 & 0 & 10 \\ -3 & 0 & 3 \end{matrix} \rightarrow G_{x}$$
y方向梯度算子
$$\begin{matrix}-3 & -10 & -3 \\ 0 & 0 & 0 \\ 3 & 10 & 3 \end{matrix}\rightarrow G_{y}$$
```
cv2.Scharr(img,ddepth,dx,dy,scale,delta(对结果偏值),borderType(BORDER_DEFAULT))
```
### Laplacian 算子
- 它的基本思想是当邻域的中心像素灰度低于它所在邻域内的其他像素的平均灰度时，此中心像素的灰度应该进一步降低；当高于时进一步提高中心像素的灰度，从而实现图像锐化处理。
$$\Delta src = \frac{\partial ^2{src}}{\partial x^2} + \frac{\partial ^2{src}}{\partial
y^2}$$
```
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
```
 - 在我们的最后一个示例中，输出数据类型为 cv.CV_8U 或 np.uint8 。但这有一个小问题。黑色到
白色的过渡被视为正斜率（具有正值），而白色到黑色的过渡被视为负斜率（具有负值）。因
此，当您将数据转换为np.uint8时，所有负斜率均设为零。简而言之，您会错过这一边缘信息。
如果要检测两个边缘，更好的选择是将输出数据类型保留为更高的形式，例如 cv.CV_16S ，cv.CV_64F 等，取其绝对值，然后转换回 cv.CV_8U 。
```
sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
```
## 4_7_Canny边缘检测
1. 降噪，5x5高斯滤波
2. 使用Sobel核在水平和垂直方向上对平滑的图像进行滤波.渐变方向始终垂直于边缘。将其舍入为代表垂直，水平和两个对角线方向的四个角度之一
$$Edge\_Gradient \; (G) = \sqrt{G_x^2 + G_y^2} \\ Angle \; (\theta) = \tan^{-1} 
\bigg(\frac{G_y}{G_x}\bigg)$$
3. 非极大值抑制：在每个像素处，检查像素是否是其在梯度方向上附近的局部最大值
4. 磁滞阈值：强度梯度大于 maxVal 的任何边缘必定是边缘，而小于 minVal 的那些边缘必定是非边缘，因此将其丢弃。介于这两个阈值之间的对象根据其连通性被分类为边缘或非边缘。如果将它们连接到“边缘”像素，则将它们视为边缘的一部分。否则，它们也将被丢弃
```
#:sobel内核大小
cv2.Canny(img,minVal,maxVal,perture_size=3,L2gradient)
```
## 4_8_图像金字塔
- 我们将需要创建一组具有不同分辨率的相同图像，并在所有图像中搜索对象。这些具有不同分辨率的图像集称为“图像金字塔”（因为当它们堆叠在底部时，最高分辨率的图像位于顶部，最低分辨率的图像位于顶部时，看起来像金字塔）
### 高斯金字塔\拉普拉斯金字塔
```
cv.pyrDown(img)
cv.pyrUp()
```
- 金字塔的一种应用是图像融合
## 4_9_1_OpenCV中的轮廓
- 为了获得更高的准确性，请使用二进制图像。因此，在找到轮廓之前，请应用阈值或canny边
缘检测。
- 要找到的对象应该是白色，背景应该是黑色
```
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
```
### 绘制轮廓
```
# k = -1,绘制所有
cv.drawContours(img, contours, k, c, width)
```
- 但是在大多数情况下，以下方法会很有用
```
cnt = contours[4]
cv.drawContours(img, [cnt], 0, (0,255,0), 3)
```
cv.CHAIN_APPROX_NONE\cv.CHAIN_APPROX_SIMPLE
## 4_9_2_轮廓特征
- 特征矩cv.moments
```
cnt = contours[0]
M = cv.moments(cnt)
```
质心
$$C_x
= \frac{M_{10}}{M_{00}} 和 C_y = \frac{M_{01}}{M_{00}}$$
```
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
```
```
cv::Moments::Moments	
(
// 空间矩(10个)
double 	m00,double 	m10,double 	m01,double 	m20,double 	m11,double 	m02,double 	m30,double 	m21,double 	m12,double 	m03  
// 中心矩（7个）
double mu20, double mu11, double mu02, double mu30, double mu21 , double mu12,double mu03
// 中心归一化矩（） 
double nu20, double nu11, double nu02, double nu30, double nu21, double nu12,double nu03;
)		
```
### 轮廓面积
cv.contourArea/或从矩 M['m00'] 中给出
```
area = cv.contourArea(cnt)
```
### 轮廓周长
```
#第二个参数指定形状是闭合轮廓( True )还是
曲线
perimeter = cv.arcLength(cnt,True)
```
### 轮廓近似
```
#epsilon:从轮廓到近似轮廓的最大距离
第三个参数指定曲线是否闭合
epsilon = 0.1*cv.arcLength(cnt,True)
cv.approxPolyDP(cnt,epsilon,True)
```
### 轮廓凸包
- 函数检查曲线是否存在凸凹缺陷并对其进行校正
```
#clockwise True输出凸包为顺时针方向。否则，其方向为逆时针方向
#returnPoints：默认情况下为True。然后返回凸包的坐标。如果为False，则返回与凸包点相对应的轮廓点的索引。
hull = cv.convexHull(points[, hull[, clockwise[, returnPoints]])
```
### 检查凸度
检查曲线是否凸出的功能
cv.isContourConvex(cnt)
### 边界矩阵
1. 直角矩形x,y,w,h = cv.boundingRect(cnt)
(x，y) 为矩形的左上角坐标，而 (w，h) 为矩形的宽度和高度
2. 旋转矩阵
- rect=中心(x,y)，(宽度，高度)，旋
转角度=cv.minAreaRect(cnt)
- cv.boxPoints(rect)得到矩阵四个角
### 最小闭合圈
(x,y),radius = cv.minEnclosingCircle(cnt)
### 拟合一个椭圆
内接椭圆的旋转矩形= ellipse = cv.fitEllipse(cnt)
### 拟合直线
[vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
## 4_9_3_轮廓属性
### 长宽比 aspect_ratio = float(w)/h
### 范围
- 轮廓区域与边界矩形区域的比值
```
area = cv.contourArea(cnt)
x,y,w,h = cv.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area
```
### 坚实度
等高线面积与其凸包面积之比
```
area = cv.contourArea(cnt)
hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area)/hull_area
```
### 等效直径
面积与轮廓面积相同的圆的直径
```
area = cv.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)
```
### 取向
物体指向的角度。以下方法还给出了主轴和副轴的长度
```
(x,y),(MA,ma),angle = cv.fitEllipse(cnt)
```
### 掩码和像素点
构成该对象的所有点
```
mask = np.zeros(imgray.shape,np.uint8)
cv.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv.findNonZero(mask)
```
### 最大值，最小值和它们的位置
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(imgray,mask = mask)
### 平均颜色或平均强度
mean_val = cv.mean(im,mask = mask)
### 极端点
- 极点是指对象的最顶部，最底部，最右侧和最左侧的点。
 ```
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
```
## 4_9_4_轮廓：更多属性
### 凸性缺陷
```
#必须在发现凸包时,传递 returnPoints= False ,以找到凸性缺陷
hull = cv.convexHull(cnt,returnPoints = False)
defects = cv.convexityDefects(cnt,hull)
它返回一个数组，其中每行包含这些值—[起点、终点、最远点、到最远点的近似距离]
```
### 点多边形测试
- 找出图像中一点到轮廓线的最短距离。它返回的距离，点在轮廓线外时为负，点在轮廓
线内时为正，点在轮廓线上时为零。
```
dist = cv.pointPolygonTest(cnt,(50,50),True)
#measureDist。如果它是真的，它会找到有符号的距离。如果为假，则查找该点是在轮廓线内部还是外部(分别返回+1、-1和0)。
```
### 形状匹配
cv.matchShapes(cnt1,cnt2,1,0.0)
## 4_9_5_轮廓分层
[Next, Previous, First_Child, Parent]
- 轮廓检索模式
### RETR_LIST
只是检索所有的轮廓，但不创建任何亲子关
系
### RETR_EXTERNAL
只返回极端外部标志。所有孩子的轮廓都被留下了。我们可以说，根据这项规则，每个家庭只有长子得到关注。它不关心家庭的其他成员
### RETR_CCOMP
所有轮廓并将其排列为2级层次结构。物体的外部轮廓(即物体的边界)放在层次结构-1
中。对象内部孔洞的轮廓(如果有)放在层次结构-2中
### RETR_TREE
## 4_10_1_直方图-1：查找，绘制，分析
- BINS(histsize) 区间
- DIMS：为其收集数据的参数的数量。
- RANGE 测量的强度值的范围。通常，它是 [0,256]
```
#:images:是uint8或float32类型的源图像,[img]
#channels:计算直方图的通道的索引,对于彩色图像，您可以传递[0]，[1]或[2]分别计算蓝色，绿色或红色通道的直方图
#histSize:全尺寸，我们通过[256]
$$cv.calcHist(images，channels，mask，histSize，ranges)
hist = cv.calcHist([img],[0],None,[256],[0,256])
hist是256x1的数组，每个值对应于该图像中具有相应像素值的像素数
```
- numpy 中直方图计算np.histogram
```
hist,bins = np.histogram(img.ravel(),256,[0,256])
#hist与我们之前计算的相同。但是bin将具有257个元素Numpy计算出bin的范围为 0-0.99 、
1-1.99 、 2-2.99 等。因此最终范围255-255.99
# 比np.histogram()快10倍左右
hist = np.bincount(img.ravel(),minlength = 256)
# OpenCV函数比np.histogram()快大约40倍
```
### 绘制直方图
1. matplotlib
```
# 1直接找到直方图并将其绘制
plt.hist(img.ravel(),256,[0,256])
# 2matplotlib的法线图
plt.plot(histr,color = col)
plt.xlim([0,256])
```
## 4_10_2_直方图-2：直方图均衡
- 另一个重要的特征是，即使图像是一个较暗的图像(而不是我们使用的一个较亮的图像)，经过均衡后，我们将得到几乎相同的图像。因此，这是作为一个“参考工具”，使所有的图像具有相同的照明条件。这在很多情况下都很有用
```
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]
```
- OpenCV中的直方图均衡
```
equ = cv.equalizeHist(img)
```
### CLAHE（对比度受限的自适应直方图均衡）
图像被分成称为“tiles”的小块，像往常一样对这些块中的每一个进行直方图均衡
- 任何直方图bin超出指定的对比度限制（在OpenCV中默认为40），则在应用直方图均衡之前，将这些像素裁剪并均匀地分布到其他bin
```
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
```
## 4_10_3_直方图3：二维直方图
- 需要将图像从BGR转换为HSV
- channel = [0,1]，因为我们需要同时处理H和S平面。
- bins = [180,256] 
- range = [0,180,0,256] 
```
hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
```
- numpy np.histogram2d
```
hist, xbins, ybins = np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,
256]])
```
### 绘制二维直方图
```
hist = cv.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
plt.imshow(hist,interpolation = 'nearest')
```
## 4_10_4_直方图-4：直方图反投影
```
# 计算对象的直方图
roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
# 直方图归一化并利用反传算法
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
# 用圆盘进行卷积
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(dst,-1,disc,dst)
# 应用阈值作与操作
ret,thresh = cv.threshold(dst,50,255,0)
thresh = cv.merge((thresh,thresh,thresh))
```
- 它创建的图像大小与输入图像相同（但只有一个通道），其中每个像素对应于该像素属于我们物体的概率.与其余部分相比，输出图像将在可能有对象的区域具有更多的白色值
## 4_11_傅里叶变换
### numpy 中
```
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
```
- 用尺寸为60x60的矩形窗口遮罩即可消除低频**np.fft.ifftshift**()应用反向移位.使用**np.ifft2**()函数找到逆FFT.同样，结果将是一个复数。您可以采用其绝对值。
### OpenCV中的傅里叶变换
**cv.dft**()和**cv.idft**()
返回与前一个相同的结果，但是有两个通道。第一个通道是结果的实部，第二个通道是结果的虚部。输入图像首先应转换为 np.float32
```
dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
```
```
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])
```
### 性能优化
对于某些数组尺寸，DFT的计算性能较好
```
nrows = cv.getOptimalDFTSize(rows)
ncols = cv.getOptimalDFTSize(cols)
# nimg = np.zeros((nrows,ncols))
# nimg[:rows,:cols] = img
right = ncols - cols
bottom = nrows - rows
nimg = cv.copyMakeBorder(img,0,bottom,0,right,bordertype, value = 0)
```
找到最优的大小
## 4_12_模板匹配
1. cv.matchTemplate返回一个灰度图像，其中每个像素表示该像素的邻域与模板匹配的程度图像的大小将为(W-w + 1，H-h + 1)
2. 可以使用**cv.minMaxLoc**()函数查找最大/最小值在哪
```
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
```
## 4_13_霍夫线变换
函数**cv.HoughLines**()返回一个：math:(rho，theta) 值的数组
- 输入图像应该是二进制图像，因此在应用霍夫变换之前，请应用阈值或使用Canny边缘检测。
- 第二和第三参数分别是ρ和θ精度。第四个参数是阈值，这意味着应该将其视为行的最低投票。请记住，票数取决于线上的点数。因此，它表示应检测到的最小线长。
```
edges = cv.Canny(gray,50,150,apertureSize = 3)
lines = cv.HoughLines(edges,1,np.pi/180,200)
for line in lines:
 rho,theta = line[0]
 a = np.cos(theta)
 b = np.sin(theta)
 x0 = a*rho
 y0 = b*rho
 x1 = int(x0 + 1000*(-b))
 y1 = int(y0 + 1000*(a))
 x2 = int(x0 - 1000*(-b))
 y2 = int(y0 - 1000*(a))
 cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
```
### 概率霍夫变换
- minLineLength - 最小行长。小于此长度的线段将被拒绝。 - maxLineGap - 线段之间允许将它们视为一条线的最大间隙
```
edges = cv.Canny(gray,50,150,apertureSize = 3)
lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
for line in lines:
 x1,y1,x2,y2 = line[0]
 cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
```
## 4_14_霍夫圈变换
```
cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
for i in circles[0,:]:
 # 绘制外圆
 cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
 # 绘制圆心
 cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
```
## 4_15_图像分割与分水岭算法
```
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# 噪声去除
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
# 确定背景区域
sure_bg = cv.dilate(opening,kernel,iterations=3)
# 寻找前景区域
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# 找到未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
# 类别标记
ret, markers = cv.connectedComponents(sure_fg)
# 为所有的标记加1，保证背景是0而不是1
markers = markers+1
# 现在让所有的未知区域为0
markers[unknown==255] = 0
#使用分水岭算法
markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]
```
## 4_16_交互式前景提取使用GrabCut算法
```
#背景，前景或可能的背景/前景0,1,2,3
# rect - 它是矩形的坐标，其中包括前景对象
#model - 应该是**cv.GC_INIT_WITH_RECT**或**cv.GC_INIT_WITH_MASK**或两者结合
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,450,290)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()
```
```
＃newmask是我手动标记过的mask图像
newmask = cv.imread('newmask.png',0)
# 标记为白色（确保前景）的地方，更改mask = 1
# 标记为黑色（确保背景）的地方，更改mask = 0
mask[newmask == 0] = 0
mask[newmask == 255] = 1
mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,
5,cv.GC_INIT_WITH_MASK)
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]
```
# 特征检测与描述
## 5_1_理解特征
- 拐角:寻找图像中在其周围所有区域中移动（少量）变化最大的区域
## 5_2_哈里斯角检测
$$E(u,v) = \sum_{x,y} \underbrace{w(x,y)}_\text{window function} \, [\underbrace{I(x+u,y+v)}
_\text{shifted intensity}-\underbrace{I(x,y)}_\text{intensity}]^2$$
$$E(u,v) \approx \begin{bmatrix} u & v \end{bmatrix} M \begin{bmatrix} u \\ v \end{bmatrix}$$
$$M = \sum_{x,y} w(x,y) \begin{bmatrix}I_x I_x & I_x I_y \\ I_x I_y & I_y I_y \end{bmatrix}$$
$$R = det(M) - k(trace(M))^2$$
其中
$$det(M)=\lambda_1\lambda_2
trace(M)=\lambda_1+\lambda_2$$
$$\lambda_1 and \lambda_2 是 M 的特征值$$
I_x和I_y分别是在x和y方向上的图像导数
1. 绝对值R较小，平坦
2. R<0，边
3. R很大，为角

数**cv.cornerHarris()**。其参数为： -img - 输入图像，应为灰度和float32类型。 - blockSize - 是拐角检测考虑的邻域大小 - ksize - 使用的Sobel导数的光圈参数。 - k - 等式中的哈里斯检测器自由参数。
```
dst = cv.cornerHarris(gray,2,3,0.04)
#result用于标记角点，并不重要
dst = cv.dilate(dst,None)
#最佳值的阈值，它可能因图像而异。
img[dst>0.01*dst.max()]=[0,0,255]
```
### SubPixel精度的转角cv.cornerSubPix
```
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
dst = cv.dilate(dst,None)
ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
#寻找质心
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
#定义停止和完善拐角的条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
```
- 进一步细化了以亚像素精度检测到的角落
## 5_3_Shi-Tomasi拐角探测器和良好的跟踪功能
$$R = min(\lambda_1, \lambda_2),如果大于阈值，则将其视为拐角。$$
```
#质量级别，该值是介于 0-1 之间的值，该值表示每个角落都被拒绝的最低拐角质量
corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
```
- 此功能更适合跟踪
## 5_4_SIFT（尺度不变特征变换）简介
- Harris拐角不是尺度不变的,但是是旋转不变的
1. 尺度空间极值检测
2. 关键点定位
3. 方向分配
4. 关键点描述
5. 关键点匹配
```
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints.jpg',img)
```
- cv.drawKeyPoints
该函数在关键点的位置绘制小圆圈。 如果将标志
**cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS**传递给它，它将绘制一个具有关键点大小的圆，甚至会显示其方向
```
cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```
### 计算描述符
```
#根据我们找到的关键点来计算描述符
kp，des = sift.compute(gray，kp)
#在单步骤中直接找到关键点和描述符
sift.detectAndCompute(img,mask)
```
## 5_5_SURF简介（加速的强大功能）
- Lowe用高斯差近似高斯的拉普拉斯算子来寻找尺度空间。SURF走得更远，使用BoxFilter近似LoG。
```
img = cv.imread('fly.png',0)
surf = cv.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(img,None)
```
## 5_6_用于角点检测的FAST算法
```
img = cv.imread('simple.jpg',0)
# 用默认值初始化FAST对象
fast = cv.FastFeatureDetector_create()
# 寻找并绘制关键点
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
```
## 5_7_BRIEF（二进制的鲁棒独立基本特征）
```
img = cv.imread('simple.jpg',0)
# 初始化FAST检测器
star = cv.xfeatures2d.StarDetector_create()
# 初始化BRIEF提取器
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# 找到STAR的关键点
kp = star.detect(img,None)
# 计算BRIEF的描述符
kp, des = brief.compute(img, kp)
print( brief.descriptorSize() )
print( des.shape )
```
## 5_8_ORB（定向快速和旋转简要）
```
img = cv.imread('simple.jpg',0)
# 初始化ORB检测器
orb = cv.ORB_create()
# 用ORB寻找关键点
kp = orb.detect(img,None)
# 用ORB计算描述符
kp, des = orb.compute(img, kp)
# 仅绘制关键点的位置，而不绘制大小和方向
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
```
## 5_9_特征匹配
### Brute-Force匹配器的基础
```
#normType:cv.NORM_L2,cv.NORM_HAMMING,cv.NORM_HAMMING2
cv.BFMatcher(normType,crossCheck)
BFMatcher.match()
BFMatcher.knnMatch()
cv.drawMatches()
cv.drawMatchesKnn()
```
```
img1 = cv.imread('box.png',cv.IMREAD_GRAYSCALE) # 索引图像
img2 = cv.imread('box_in_scene.png',cv.IMREAD_GRAYSCALE) # 训练图像
# 初始化ORB检测器
orb = cv.ORB_create()
# 基于ORB找到关键点和检测器
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True) # 匹配描述符.
matches = bf.match(des1,des2) # 根据距离排序
matches = sorted(matches, key = lambda x:x.distance) # 绘制前10的匹配项
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:
10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```
函数返回是DMatch对象的列表,DMatch有以下属性
- DMatch.distance:描述符之间的距离
- DMatch.trainIdx:火车描述符中的描述符索
引
- DMatch.imgIdx-火车图像的索引
### 基于匹配器的FLANN
```
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
 table_number = 6, # 12
 key_size = 12, # 20
 multi_probe_level = 1) #2

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
```
## 5_10_特征匹配+单应性查找对象
在复杂图像中找到已知对象。
```
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)

dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

matchesMask = mask.ravel().tolist()
h,w,d = img1.shape

pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv.perspectiveTransform(pts,M)
img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
draw_params = dict(matchColor = (0,255,0), # 用绿色绘制匹配
 singlePointColor = None,
 matchesMask = matchesMask, # 只绘制内部点
 flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
```
# 视频分析
## 6_1_如何使用背景分离方法
- 背景分离（BS）是一种通过使用静态相机来生成前景掩码（即包含属于场景中的移动对象像素的二进制图像）的常用技术
```
#创建背景分离对象
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
```
每帧都用于计算前景掩码和更新背景
```
fgMask = backSub.apply(frame)
```
## 6_2_Meanshift和Camshift
跟踪视频中对象.设置目标，找到其直方图，以便我们可以将目标反投影到每帧上以计算均值偏移。我们还需要提供窗口的初始位置
### Meanshift
初始化
```
track_window = (x, y, w, h)
#设置初始ROI来追踪
roi = frame[y:y+h, x:x+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
#设置终止条件，可以是10次迭代，也可以至少移动1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
```
更新
```
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
#应用meanshift来获取新位置
ret, track_window = cv.meanShift(dst, track_window, term_crit)
```
### Camshift
会更新窗口的大小
## 6_3_光流
光流是由物体或照相机的运动引起的两个连续帧之间图像物体的视运动的模式。它是2D向量场，
其中每个向量都是位移向量，表示点从第一帧到第二帧的运动。
### Lucas-Kanade
初始化
```
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#创建用于作图的掩码图像
mask = np.zeros_like(old_frame)
```
更新
```
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#计算光流
p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
**lk_params)
```
### Gunner Farneback的算法
- 查找密集的光流

初始化
```
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
```
更新
```
next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
prvs = next
```
# 相机校准和3D重建
## 7_1_相机校准
径向变形
$$x_{distorted} = x( 1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \\ y_{distorted} = y( 1 + k_1 r^2 + k_2 r^4 + k_3
r^6)$$
切向畸变
$$x_{distorted} = x + [ 2p_1xy + p_2(r^2+2x^2)] \\ y_{distorted} = y + [ p_1(r^2+ 2y^2)+ 2p_2xy]$$
需要找到五个参数，称为失真系数
$$Distortion \; coefficients=(k_1 \hspace{10pt} k_2 \hspace{10pt} p_1 \hspace{10pt} p_2
\hspace{10pt} k_3)$$
焦距和光学中心可用于创建相机矩阵，该相机矩阵可用于消除由于特定相机镜头而引起的畸变.
焦距(f_x，f_y)和光学中心(c_x,c_y)
$$camera \; matrix = \left [ \begin{matrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{matrix}
\right ]$$
要找到这些参数，我们必须提供一些定义良好的图案的示例图像.至少需要10个测试模式

使用cv.findChessboardCorners找棋盘角落
```
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 #找到棋盘角落
ret, corners = cv.findChessboardCorners(gray, (7,6), None)
 #如果找到，添加对象点，图像点（细化之后）
if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
```
使用函数**cv.calibrateCamera**()
返回相机矩阵，失真系数，旋转和平移矢量等。
```
objpoints = [] # 真实世界中的3d点
imgpoints = [] # 图像中的2d点
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints,
gray.shape[::-1], None, None)
```
优化相机矩阵
```
img = cv.imread('left12.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
```
### 拍摄图像并对其进行扭曲。
1. 使用cv.undistort()
```
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
```
2. 使用remapping
```
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
```
### 重投影误差
估计找到的参数的精确程度
```
for i in xrange(len(objpoints)):
imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
mean_error += error
```
## 7_2_姿态估计
## 7_3_对极几何
## 7_4_立体图像的深度图
# 机器学习
## 8_1_理解KNN
## 8_2_使用OCR手写数据集运行KNN
## 8_3_理解SVM
## 8_4_使用OCR手写数据集运行SVM
## 8_5_理解K均值聚类
## 8_6_OpenCV中的K均值
# 计算摄影学
## 9_1_图像去噪
## 9_2_图像修补
## 9_3_高动态范围
# 目标检测
## 10_1_级联分类器
## 10_2_级联分类器训练