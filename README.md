## ——————CSTDetection(CNN+网格预测)——————

### ——项目简介——

        一个从零实现的多目标检测器，采用了YOLOv1风格的网格预测思想，专用于检测CST数据集中的圆形（Circle）、矩形（Square）、三角形（Triangle）三个类别。

### ——数据集简介——

--数据集名称：CST

--地址：https://github.com/Geeshang/cst-dataset

--类别：0-圆形，1-矩形，2-三角形

--大小：训练集600张，验证集100张，测试集200张

--标注格式：YOLO（x_c, y_c, w, h）

### ——模型信息——

--输入：[Batch_size,3,448,448]

--输出：[Batch_size,7,7,13]

--网格划分：7×7（C=7）

--每个网格预测框数：2（B=2）

--结构：

1、特征提取模块：采用MobileNetV2的特征提取层（无预训练）。

![](./README_imgs/1.png)

2、特征处理模块：采用Conv+BN+LeakyReLU的叠加来下采样和细化特征。

<img src="./README_imgs/2.png" title="" alt="" width="709">

### ——性能结果——

--评估指标：若类别预测准确且IOU>0.5则判断预测准确，预测准确率correct_rate=预测准确数/目标总数

--训练轮次：100

--损失函数：位置损失(MSELoss)+置信度损失(MSELoss)+类别损失(CrossEntropyLoss)

--优化器：Adam（lr=0.001）

--最佳准确率：92%（测试集）

### ——预测示例——

--此示例存放于detect目录下：

<img title="" src="./README_imgs/3.png" alt="" width="703">           
