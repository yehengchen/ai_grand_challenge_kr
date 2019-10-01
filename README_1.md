# 运行环境
* 硬件配置: GTX 1080Ti x 1 
* 系统版本: Ubuntu 16.04
* 开发语言: Python 3.5

# 运行脚本

1.  环境部署脚本：pip3 install -r requ

2.  一键训练脚本：./train.sh

3.  一键测试脚本：./test.sh


# 详细的运行说明

## 算法原理: 算法原理.pdf

## 代码原码: main.py

## 模型:yolov3-spp.h5

## 数据或数据链接

# 代码运行说明:

1.  各脚本的执行步骤，模型及提交结果的保存路径:

a. 下载训练数据并解压图片及标记文件至 train/ 文件夹下，执行 ./train.sh 最终模型保存至 train/backup/ 。

b. 训练完成后执行 ./convert.sh 将模型转换后，执行 ./test.sh 对 test_video/ 目录下的测试集进行测试并输出结果txt文件。

2.  运行时间: 4 * 15 min = 60 min

# 相关模型训练详细说明

训练模型需要并修改 ./train/yolo3_object.data 的路径后执行 ./train.sh 进行训练
检测模型基于 darknet 框架，训练集提取了VOC2007和2012、COCO数据集的行人数据和虚拟行人数据集进行训练
追踪模型基于 cosine_metric_learning 训练集采用了Market1501 MOT行人数据集进行训练


行人多目标跟踪_fly_piggy
