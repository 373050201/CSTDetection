"""
模型训练
"""
from loss import YoloLoss
from model import YoloModel
from yolo_dataset import YoloDataset,create_yolo_target
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torchvision.ops import box_iou



print("正在加载训练/验证集...")
batch_size=8
train_dataset = YoloDataset("datasets/cst_yolo/images/train",
                            "datasets/cst_yolo/labels/train",
                            img_transform=transforms.Compose([
                                transforms.Resize((448, 448)),
                                transforms.ToTensor(),
                            ]),
                            label_transform=create_yolo_target)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
val_dataset = YoloDataset("datasets/cst_yolo/images/val",
                            "datasets/cst_yolo/labels/val",
                            img_transform=transforms.Compose([
                                transforms.Resize((448, 448)),
                                transforms.ToTensor(),
                            ]),
                            label_transform=create_yolo_target)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

print("正在初始化模型...")
yoloModel=YoloModel()
yoloModel=yoloModel.cuda()
yoloLoss=YoloLoss()
yoloLoss=yoloLoss.cuda()
learning_rate=0.001
optimizer=torch.optim.Adam(yoloModel.parameters(),lr=learning_rate)

total_train=0
epoch=100
best_rate=0
best_epoch=0
for i in range(epoch):
    print(f"第{i+1}轮训练开始...")
    total_train_loss=0
    for imgs,labels in train_loader:
        imgs=imgs.cuda()
        labels=labels.cuda()

        preds=yoloModel(imgs)
        trainLoss=yoloLoss(preds,labels)
        total_train_loss+=trainLoss
        optimizer.zero_grad()
        trainLoss.backward()
        optimizer.step()

        total_train+=1
    print(f"此轮已累计训练了{total_train}次")
    print(f"训练集总损失：{total_train_loss:.6f}")

    total_val_loss=0
    total_obj=0
    total_correct=0
    with torch.no_grad():
        for imgs,labels in val_loader:
            imgs=imgs.cuda()
            labels=labels.cuda()

            preds=yoloModel(imgs)
            lossVal=yoloLoss(preds,labels)
            total_val_loss+=lossVal

            for b in range(batch_size):
                for grid_y in range(7):
                    for grid_x in range(7):
                        if labels[b,grid_y,grid_x,4]>0.5:#有目标
                            total_obj+=1
                            #目标框解码
                            cls_tgt=torch.argmax(labels[b,grid_y,grid_x,10:13]).item()#目标类别
                            cls_pred = torch.argmax(preds[b,grid_y,grid_x,10:13]).item()#预测类别
                            if cls_tgt != cls_pred:#若类别预测错误，则无需后续判断
                                continue
                            x_offset_tgt=labels[b,grid_y,grid_x,0].item()
                            y_offset_tgt=labels[b,grid_y,grid_x,1].item()
                            w_tgt=labels[b,grid_y,grid_x,2].item()
                            h_tgt=labels[b,grid_y,grid_x,3].item()
                            x_c_tgt=(grid_x+x_offset_tgt)/7
                            y_c_tgt=(grid_y+y_offset_tgt)/7
                            bbox_tgt=torch.tensor([x_c_tgt,y_c_tgt,w_tgt,h_tgt])#目标框
                            bbox_tgt.unsqueeze_(0)#shape:[1,4]
                            #预测框解码
                            best_iou=0
                            for box_idx in range(2):#有2个预测框
                                start_idx=box_idx*5
                                x_offset_pred=preds[b,grid_y,grid_x,start_idx].item()
                                y_offset_pred=preds[b,grid_y,grid_x,start_idx+1].item()
                                w_pred=preds[b,grid_y,grid_x,start_idx+2].item()
                                h_pred=preds[b,grid_y,grid_x,start_idx+3].item()
                                x_c_pred=(grid_x+x_offset_pred)/7#转化为x_center/y_center
                                y_c_pred=(grid_y+y_offset_pred)/7
                                bbox_pred=torch.tensor([x_c_pred,y_c_pred,w_pred,h_pred])#预测框
                                bbox_pred.unsqueeze_(0)#shape:1*4
                                iou=box_iou(bbox_pred,bbox_tgt,fmt="cxcywh").item()
                                if iou>best_iou:#计算两个框最大的iou
                                    best_iou=iou
                            if best_iou>0.5:#若iou大于0.5，则判断预测成功
                                total_correct+=1
    print(f"验证集总损失：{total_val_loss:.6f}")
    correct_rate=total_correct/total_obj
    print(f"验证集预测准确率：{correct_rate:.6f}")
    if correct_rate>best_rate:
        best_rate=correct_rate
        best_epoch=i+1
        torch.save(yoloModel.state_dict(),f"best_model.pth")
print("训练结束")
print(f"最佳训练轮次：{best_epoch}，验证集准确率为：{best_rate:.6f}")