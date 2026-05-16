"""
损失类，用于定义损失函数
"""
import torch
from torch import nn



class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss,self).__init__()
        self.location_loss=nn.MSELoss()#位置损失
        self.confidence_loss=nn.MSELoss()#置信度损失
        self.class_loss=nn.CrossEntropyLoss()#类别损失
    def forward(self,predict,target):
    #predict/target形状:[B,7,7,13]
        lct_loss=self.location_loss(predict[...,0:4],target[...,0:4])+self.location_loss(predict[...,5:9],target[...,5:9])
        cfd_loss=self.confidence_loss(predict[...,4],target[...,4]+self.confidence_loss(predict[...,9],target[...,9]))
        cls_loss=self.class_loss(predict[...,10:13],target[...,10:13])
        sum_loss=lct_loss+cfd_loss+cls_loss
        return sum_loss



if __name__=="__main__":
    B=2
    predict=torch.randn(B,7,7,13)
    target=torch.zeros(B,7,7,13)
    loss=YoloLoss()
    lossVal=loss(predict,target)
    print(lossVal)