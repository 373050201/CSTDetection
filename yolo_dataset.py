"""
数据集类，用于访问数据集
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms



class YoloDataset(Dataset):
    def __init__(self,img_folder,label_folder,img_transform,label_transform):
        self.img_folder=img_folder
        self.label_folder=label_folder
        self.img_transform=img_transform
        self.label_transform=label_transform
        self.img_names=os.listdir(self.img_folder)
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self,idx):
        img_name=self.img_names[idx]
        img_path=os.path.join(self.img_folder,img_name)
        img=Image.open(img_path)
        label_name=img_name.split(".")[0]+".txt"
        label_path=os.path.join(self.label_folder,label_name)
        bboxes=[]
        clses=[]
        with open(label_path) as f:
            label_content=f.read()
            obj_infos=label_content.strip().split("\n")
            for obj_info in obj_infos:
                info_list=obj_info.strip().split(" ")
                cls=int(info_list[0])
                x_c=float(info_list[1])
                y_c=float(info_list[2])
                w=float(info_list[3])
                h=float(info_list[4])
                clses.append(cls)
                bboxes.append([x_c,y_c,w,h])
        label={
            "obj_num": len(bboxes),
            "clses":clses,
            "bboxes":bboxes,
        }
        label["clses"]=torch.tensor(label["clses"])
        label["bboxes"]=torch.tensor(label["bboxes"])
        if self.img_transform is not None:
            img=self.img_transform(img)
        if self.label_transform is not None:
            label=self.label_transform(label)
        return img,label
"""
默认返回的label结构：
label={
        "obj_num": #目标个数
        "clses":#类别
        "bboxes":#预测框
    }
"""
def create_yolo_target(label):#将label转化为模型训练所需格式:[7,7,13]
    S=7;B=2;C=3
    target=torch.zeros((S,S,B*5+C))#此处target形状：[7,7,13]，13包括(2*(x_offset,y_offset,w,h,conf)+3*cls)
    clses=label["clses"]
    bboxes=label["bboxes"]
    for cls,bbox in zip(clses,bboxes):
        x,y,w,h=bbox
        #网格x，y坐标(S*S坐标系中)
        grid_x=int(x*S)
        grid_y=int(y*S)
        #bbox中心在网格内的归一化坐标
        x_offset=x*S-grid_x
        y_offset=y*S-grid_y
        #简化yolo:若第一个框可用则进行填充(置信度为0则认为无目标，为1则有)
        if target[grid_y,grid_x,4]==0:
            box_idx=0
        else:
            continue
        #在对应位置填充坐标和置信度
        start_idx=box_idx*5
        target[grid_y,grid_x,start_idx]=x_offset
        target[grid_y,grid_x,start_idx+1]=y_offset
        target[grid_y,grid_x,start_idx+2]=w
        target[grid_y,grid_x,start_idx+3]=h
        target[grid_y,grid_x,start_idx+4]=1.0#置信度置1，标记为有目标
        target[grid_y,grid_x,B*5+cls]=1.0#目标类别置1，onehot编码
    return target



if __name__=="__main__":
    train_dataset=YoloDataset("datasets/cst_yolo/images/train",
                              "datasets/cst_yolo/labels/train",
                              img_transform=transforms.Compose([
                                  transforms.Resize((448,448)),
                                  transforms.ToTensor(),
                              ]),
                              label_transform=create_yolo_target)
    print(len(train_dataset))
    img,label=train_dataset[0]
    print(img.shape)
    print(label.shape)