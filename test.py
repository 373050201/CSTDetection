"""
模型测试
"""
from model import YoloModel
from yolo_dataset import YoloDataset,create_yolo_target,create_cls_list
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torchvision.ops import box_iou
import random
import os
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt



print("正在加载测试集...")
batch_size=8
test_dataset=YoloDataset("datasets/cst_yolo/images/test",
                         "datasets/cst_yolo/labels/test",
                         img_transform=transforms.Compose([
                             transforms.Resize((448, 448)),
                             transforms.ToTensor(),
                         ]),
                         label_transform=create_yolo_target)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

print("正在加载模型...")
yoloModel=YoloModel()
yoloModel=yoloModel.cuda()
yoloModel.load_state_dict(torch.load("./best_model.pth"))

print("开始测试...")
total_obj=0
total_correct=0
for imgs,labels in test_loader:
    imgs=imgs.cuda()
    labels=labels.cuda()

    preds=yoloModel(imgs)
    for b in range(batch_size):
        for grid_y in range(7):
            for grid_x in range(7):
                if labels[b,grid_y,grid_x,4]>0.5:#有目标
                    total_obj+=1
                    #目标框解码
                    cls_tgt=torch.argmax(labels[b,grid_y,grid_x,10:13]).item()
                    cls_pred=torch.argmax(preds[b,grid_y,grid_x,10:13]).item()
                    if cls_tgt!=cls_pred:#类别不同则判断预测错误，无需后续比较
                        continue
                    x_offset_tgt=labels[b,grid_y,grid_x,0].item()
                    y_offset_tgt=labels[b,grid_y,grid_x,1].item()
                    w_tgt=labels[b,grid_y,grid_x,2].item()
                    h_tgt=labels[b,grid_y,grid_x,3].item()
                    x_c_tgt=(grid_x+x_offset_tgt)/7
                    y_c_tgt=(grid_y+y_offset_tgt)/7
                    bbox_tgt=torch.tensor([x_c_tgt, y_c_tgt, w_tgt, h_tgt])#目标框
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

correct_rate=total_correct/total_obj
print(f"测试集预测准确率：{correct_rate:.6f}")

show_num=3#随机展示show_num张图片
for i in range(show_num):
    idx=random.randint(0,len(test_dataset)-1)
    img,_=test_dataset[idx]
    img.unsqueeze_(0)
    img=img.cuda()
    pred=yoloModel(img)
    pred=pred[0]
    img_name="cst-"+str(700+idx)+".jpg"
    img_path=os.path.join("datasets/cst_yolo/images/test",img_name)
    img=Image.open(img_path)#原始图片
    W,H=img.size#原始图片宽高

    for grid_y in range(7):
        for grid_x in range(7):
            for box_idx in range(2):
                start_idx=box_idx*5
                conf=pred[grid_y,grid_x,start_idx+4]#置信度
                cls=torch.argmax(pred[grid_y,grid_x,10:13],dim=0).item()#类别
                cls_list=create_cls_list("cst_yolo")
                if conf>0.5:#若置信度大于0.5，则画框
                    #解码bbox
                    x_offset=pred[grid_y,grid_x,start_idx].item()
                    y_offset=pred[grid_y,grid_x,start_idx+1].item()
                    w=pred[grid_y,grid_x,start_idx+2].item()
                    h=pred[grid_y,grid_x,start_idx+3].item()
                    # 转化为(x_min,y_min)与(x_max,y_max)
                    x_center=(grid_x+x_offset)/7
                    y_center=(grid_y+y_offset)/7
                    x_max=W*(x_center+0.5*w)
                    y_max=H*(y_center+0.5*h)
                    x_min=W*(x_center-0.5*w)
                    y_min=H*(y_center-0.5*h)
                    draw=ImageDraw.Draw(img)#创建Draw类的对象
                    draw.rectangle(xy=[x_min,y_min,x_max,y_max],outline="green",width=2)#绘制矩形
                    draw.text(xy=(x_min,y_min-25),text=f"{cls_list[cls]}",fill="green",font_size=20)#绘制预测类别

    plt.imshow(img)#可视化图片
    plt.axis("off")
    plt.show()