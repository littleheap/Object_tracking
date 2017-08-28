# Object Tracking

目标跟踪是对摄像头视频中的移动目标进行定位的过程，有着非常广泛的应用。实时目标跟踪是许多计算机视觉应用的重要任务，如监控、基于感知的用户界面、增强现实、基于对象的视频压缩以及辅助驾驶等。

有很多实现视频目标跟踪的方法，当跟踪所有移动目标时，帧之间的差异会变的有用；当跟踪视频中移动的手时，基于皮肤颜色的均值漂移方法是最好的解决方案；当知道跟踪对象的一方面时，模板匹配是不错的技术。


 **- Basic_object_detection.py** 

是做一个基本的运动检测，考虑的是“背景帧”与其它帧之间的差异，这种方法检测结果还是挺不错的，但是需要提前设置背景帧，如果是在室外，光线的变化就会引起误检测，还是很有局限性的。
![输入图片说明](https://git.oschina.net/uploads/images/2017/0823/102724_44d07c58_1487586.png "20170621092612303.png")


OpenCV提供了一个称为BackgroundSubtractor的类，在分割前景和背景时很方便。 
在OpenCV3中有三种背景分割器：K-Nearest（KNN）、Mixture of Gaussians（MOG2）、Geometric Multigid（GMG）

BackgroundSubtractor类是专门用于视频分析的，即BackgroundSubtractor类会对每帧的环境进行“学习”。BackgroundSubtractor类常用来对不同帧进行比较，并存储以前的帧，可按时间推移方法来提高运动分析的结果。

 **- Background_splitter_MOG2.py** 

![输入图片说明](https://git.oschina.net/uploads/images/2017/0824/081017_95859534_1487586.png "20170621173838645.png")

BackgroundSubtractor类的另一个基本特征是它可以计算阴影。这对于精确读取视频帧绝对是至关重要的；通过检测阴影，可以排除检测图像的阴影区域（采用阈值方式），从而能关注实际特征。

 **- Background_splitter_KNN.py** 

![输入图片说明](https://git.oschina.net/uploads/images/2017/0824/081106_97b09c7a_1487586.png "20170621180534284.png")

(图片从左到右依次为：检测出的运动目标、背景分割、背景分割后阈值化)

 **- Kalman_mouse_tracking.py** 

卡尔曼是匈牙利数学家，Kalman滤波器源于其博士毕业了论文和1960年发表的论文《A New Approach to Linear Filtering and Prediction Problems》（线性滤波与预测问题的新方法）。[论文地址](http://xueshu.baidu.com/s?wd=paperuri:%2885cb47b4792381a0e07affaf64865747%29&filter=sc_long_sign&sc_ks_para=q=A%20New%20Approach%20to%20Linear%20Filtering%20and%20Prediction%20Problems&tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_us=17845477967171964869) 
1. 卡尔曼滤波在很多领域都得到了应用，特别在飞机、导弹等导航制导方面经常用到。
2. 卡尔曼滤波器会对含有噪声的输入数据流（比如计算机视觉中的视频输入）进行递归操作，并产生底层系统状态（比如视频中的位置）在统计意义上的最优估计。
3. 这里是列表文本卡尔曼滤波算法分为两个阶段：
    - 预测阶段：卡尔曼滤波器使用由当前点计算的协方差来估计目标的新位置；
    - 更新阶段：卡尔曼滤波器记录目标的位置，并为下一次循环计算修正协方差。

整个卡尔曼滤波的过程就是个递推计算的过程，不断的“预测–更新–预测–更新……”
![输入图片说明](https://git.oschina.net/uploads/images/2017/0824/082414_be0bb08a_1487586.png "20170704103159798.png")

**- Face_capture.py**

【Python+OpenCV】实现检测场景内是否有物体移动，并进行人脸检测抓拍
![输入图片说明](http://img.blog.csdn.net/20170608200539380?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHdwbHdm/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


