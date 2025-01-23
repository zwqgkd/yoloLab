import os
import random
import shutil
import torch


"""
统计类别个数
"""
def get_class_id(labelPath):
    class_map ={}
    for txtFile in os.listdir(labelPath):
        if 'snow' in str(txtFile):
            continue
        with open(os.path.join(labelPath, txtFile)) as f:
            lines = f.readlines()
            for line in lines:
                class_id =line.split()[0]
                if class_id not in class_map:
                    class_map[class_id]=1
                else:
                    class_map[class_id]+=1
    return class_map

"""
delete id
"""
def delete_id(labelPath, idList):
    for txtFile in os.listdir(labelPath):
        with open(os.path.join(labelPath, txtFile), "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                if int(parts[0]) in idList:
                    lines.remove(line)
        with open(os.path.join(labelPath, txtFile), "w") as f:
            f.writelines(lines)



"""重新给class_id编号"""
def renumber_class_ids(labels_path,class_id_mapping):
    snow_label_count=0
    print("类别 ID 映射：", class_id_mapping)

    for label_file in os.listdir(labels_path):
        if('snow' in str(label_file)):
            snow_label_count+=1
            continue
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
    print(snow_label_count)
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

splitPath = R"datasets/dawn_dan"

rootPath =R"C:\Projects\gkd\dataset\766ygrbt8y-3\DAWN"


# dataSetSplit(os.path.join(rootPath, "Snow", "Snow"), os.path.join(rootPath, "Snow", "Snow", "Snow_YOLO_darknet"), splitPath)



# 参考1，3，8 


# delete_id(os.path.join(splitPath, "train", "labels"), [7])



print(get_class_id(os.path.join(splitPath, "train", "labels")))
class_id_mapping ={1:0,3:1,8:2,2:3,4:4,6:5}
renumber_class_ids(os.path.join(splitPath, "train", "labels"), class_id_mapping)





