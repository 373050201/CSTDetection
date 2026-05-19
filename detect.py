"""
模型部署
"""
import torch
from PIL import Image,ImageDraw
from model import YoloModel
from torchvision import transforms
import matplotlib.pyplot as plt
from yolo_dataset import create_cls_list



def load_img(path):#加载模型输入图片
    img=Image.open(path)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((448,448))
    ])
    return transform(img)



print("正在加载模型...")
yoloModel=YoloModel()
yoloModel=yoloModel.cuda()
yoloModel.load_state_dict(torch.load("./best_model.pth"))
img_path="detect/detect.jpg"

print("正在预测...")
img=load_img(img_path)
img.unsqueeze_(0)
img=img.cuda()
pred=yoloModel(img)#shape:[1,7,7,13]
pred=pred[0]#shape:[7,7,13]
img=Image.open(img_path)#原始图片
W,H=img.size#原始图片宽高

for grid_y in range(7):
    for grid_x in range(7):
        for box_idx in range(2):
            start_idx=box_idx*5
            conf=pred[grid_y,grid_x,start_idx+4]#置信度
            cls=torch.argmax(pred[grid_y, grid_x,10:13],dim=0).item()#类别
            cls_list=create_cls_list("cst_yolo")
            if conf>0.5:#若置信度大于0.5，则画框
                #解码bbox
                x_offset=pred[grid_y,grid_x,start_idx].item()
                y_offset=pred[grid_y,grid_x,start_idx+1].item()
                w=pred[grid_y,grid_x,start_idx+2].item()
                h=pred[grid_y,grid_x,start_idx+3].item()
                #转化为(x_min,y_min)与(x_max,y_max)
                x_center=(grid_x+x_offset)/7
                y_center=(grid_y+y_offset)/7
                x_max=W*(x_center+0.5*w)
                y_max=H*(y_center+0.5*h)
                x_min=W*(x_center-0.5*w)
                y_min=H*(y_center-0.5*h)
                draw=ImageDraw.Draw(img)#创建Draw类的对象
                draw.rectangle(xy=[x_min,y_min,x_max,y_max],outline="green",width=2)#绘制矩形
                draw.text(xy=(x_min,y_min-25),text=f"{cls_list[cls]}",fill="green",font_size=20)#绘制预测类别
                draw.text(xy=(x_min,y_max),text=f"conf:{conf:.2f}",fill="red",font_size=12)#绘制置信度

img.save("detect/result.jpg")
plt.imshow(img)#可视化图片
plt.axis("off")
plt.show()