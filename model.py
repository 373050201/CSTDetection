"""
模型类，用于定义模型
"""
import torch
from torch import nn
from yolo_dataset import YoloDataset
import torchvision.transforms as transforms
import torchvision.models as models



class YoloModel(nn.Module):#yolo风格模型
    def __init__(self,S=7,B=2,C=3):
        super(YoloModel, self).__init__()
        self.S=S,#将img划分为S*S个网格
        self.B=B,#每个网格有B个bbox
        self.C=C,#类别数
        mobilenet=models.mobilenet_v2()
        # ==========1、特征提取模块==========
        self.extract=mobilenet.features#shape:[1280,14,14]
        # ==========2、特征处理模块==========
        self.process=nn.Sequential(
            nn.Conv2d(1280,512,3,2,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            #shape:[512,7,7]
            nn.Conv2d(512,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            #shape:[256,7,7]
            nn.Conv2d(256,B*5+C,1),
            #shape:[B*5+C,7,7]
        )
    def forward(self,x):
        feature=self.extract(x)
        y=self.process(feature)
        y=y.permute(0,2,3,1)
        y[...,0:10]=torch.sigmoid(y[...,0:10])#坐标和置信度用sigmoid归一
        #y[...,10:13]=torch.softmax(y[...,10:13],-1)#分类用softmax，交叉熵损失涵盖此过程
        return y



if __name__=='__main__':
    myModel=YoloModel()
    train_dataset = YoloDataset("./datasets/cst_yolo/images/train",
                                "./datasets/cst_yolo/labels/train",
                                img_transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((448, 448)),
                                ]),
                                label_transform=None)
    img,label=train_dataset[0]
    img.unsqueeze_(0)
    output=myModel(img)
    print(output.shape)