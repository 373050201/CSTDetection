"""
将官网下载的CST数据集转换为自己需要的目录格式，并将原标注转换为yolo格式标注
CST数据集官网：https://github.com/Geeshang/cst-dataset
原始CST数据集即cst_dataset，直接放在项目根目录，train:val:test=600:100:300
"""
import shutil
import os
import csv



def copy_img():#将图片从原始数据集复制到对应目录
    print("复制图片中...")
    head_list=["train","val","test"]#train:0-599,val:600-699,test:700-999
    for head in head_list:
        source_folder=os.path.join("cst-dataset",head)
        length=len(os.listdir(source_folder))
        for i in range(length):
            if head=="val":#偏移量
                i+=600
            elif head=="test":
                i+=700
            img_folder_name="cst-"+str(i)
            img_name="cst-"+str(i)+".jpg"
            img_path=os.path.join(source_folder,img_folder_name,img_name)#原始图片路径
            target_path=os.path.join("datasets","cst_yolo","images",head,img_name)#目标路径/新图片名
            shutil.copy(img_path,target_path)
    print("图片复制完成")



def trans_label():#将原标注转化为yolo格式标注并写入对应目录
    print("解析标注中...")
    head_list=["train","val","test"]
    source_folder=os.path.join("cst-dataset","annotation")
    for head in head_list:
        label_name="anno-"+head+".csv"
        label_path=os.path.join(source_folder,label_name)#原始标签路径
        target_folder=os.path.join("datasets","cst_yolo","labels",head)#目标文件夹路径
        with open(label_path,"r") as f:
            reader=csv.reader(f)
            next(reader)#跳过标题行
            for row in reader:
                img_name=row[0]+".txt"#读取图片名，令标注与图片同名
                target_path=os.path.join(target_folder,img_name)
                cls=row[2]#解析cls为类别索引
                if cls=='c':
                    cls=0
                elif cls=='s':
                    cls=1
                elif cls=='t':
                    cls=2
                bbox=row[3]#解析bbox为yolo格式
                bbox_list=bbox.strip().split(" ")
                img_size=512
                W=img_size
                H=img_size
                x_min=float(bbox_list[0])
                y_min=float(bbox_list[1])
                x_max=float(bbox_list[2])
                y_max=float(bbox_list[3])
                x_c=(x_max+x_min)/(2*W)
                y_c=(y_max+y_min)/(2*H)
                w=(x_max-x_min)/W
                h=(y_max-y_min)/H
                print(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")#yolo格式：(x_c,y_c,w,h)
                with open(target_path,'a') as f:
                    f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
    print("标注解析完成")



if __name__ == "__main__":
    print("正在处理原始数据集...")
    #copy_img()
    #trans_label()