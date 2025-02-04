import os
import random
import shutil
import torch


"""
统计类别个数
"""
def get_class_ids(labels_path):
    class_ids={}
    for label_file in os.listdir(labels_path):
        if label_file.endswith(".txt"):
            try:
                with open(os.path.join(labels_path, label_file), "r") as f:
                    for line in f:
                        id=int(line.split(" ")[0])
                        if id in class_ids:
                            class_ids[id]+=1
                        else:
                            class_ids[id]=1
            except IOError:
                print(f"警告：文件{label_file}无法打开")
    return class_ids

"""重新给class_id编号"""
def renumber_class_ids(labels_path,class_id_mapping):
    print("类别 ID 映射：", class_id_mapping)

    for label_file in os.listdir(labels_path):
        if(label_file.endswith(".txt")):
            label_file_path = os.path.join(labels_path, label_file)
            try:
                with open(label_file_path, "r") as f:
                    lines= f.readlines()
                with open(label_file_path, "w") as f:     
                    for line in lines:
                        parts=line.split(" ")
                        old_class_id = int(parts[0])
                        new_class_id = class_id_mapping[old_class_id]
                        parts[0]=str(new_class_id)
                        f.write(" ".join(parts))
            except IOError:
                print(f"警告：文件{label_file}无法打开")
"""
做数据分类
"""
def dataSetSplit(imgDataSetPath, lableDataSetPath, splitPath, splitRate=0.8):
    trainPath = os.path.join(splitPath, "train")
    valPath = os.path.join(splitPath, "val")
    if not os.path.exists(trainPath):
        os.makedirs(os.path.join(trainPath, "images"))
        os.makedirs(os.path.join(trainPath, "labels"))
    if not os.path.exists(valPath):
        os.makedirs(os.path.join(valPath, "images"))
        os.makedirs(os.path.join(valPath, "labels"))
    # split the dataset
    files_and_dirs = os.listdir(imgDataSetPath)
    files = [file for file in files_and_dirs if os.path.isfile(os.path.join(imgDataSetPath, file))]
    for file in files:
        if random.random()<splitRate:
            shutil.copy(os.path.join(imgDataSetPath, file), os.path.join(trainPath, "images"))
            fileLabel = file.split(".")[0] + ".txt"
            shutil.copy(os.path.join(lableDataSetPath, fileLabel), os.path.join(trainPath, "labels"))
        else:
            shutil.copy(os.path.join(imgDataSetPath, file), os.path.join(valPath, "images"))
            fileLabel = file.split(".")[0] + ".txt"
            shutil.copy(os.path.join(lableDataSetPath, fileLabel), os.path.join(valPath, "labels"))




# print(torch.cuda.is_available())  # 如果返回 True，说明有 GPU 可用
# print(torch.cuda.device_count())  # 返回 GPU 的数量

splitPath = R"datasets/dawn"

rootPath =R"C:\Projects\gkd\dataset\766ygrbt8y-3\DAWN"


# dataSetSplit(os.path.join(rootPath, "Snow", "Snow"), os.path.join(rootPath, "Snow", "Snow", "Snow_YOLO_darknet"), splitPath)



print(get_class_ids(os.path.join(splitPath, "train", "labels")))
print(get_class_ids(os.path.join(splitPath, "val", "labels")))
# class_id_mapping ={1:0,2:1,3:2,4:3,6:4,8:5}
# renumber_class_ids(os.path.join(splitPath, "train", "labels"), class_id_mapping)
# renumber_class_ids(os.path.join(splitPath, "val", "labels"), class_id_mapping)




