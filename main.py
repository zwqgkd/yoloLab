from ultralytics import YOLO  # 导入 YOLO 模型
import os

import matplotlib
matplotlib.use('TkAgg',force=True)

def train(yamlPath, dataPath, modelName="yolo11n.pt"):
    # 1. 加载预训练模型
    model = YOLO(yamlPath).load(modelName)  # 这里假设你已经下载并使用了一个预训练模型的路径
    # model = YOLO("yolo11n.pt")
    # 2. 训练模型
    train_results = model.train(
    data=dataPath,  # 数据集配置文件路径，这里你应该提供正确的 YAML 配置文件路径
    epochs=100,  # 训练的轮数，可以根据需要调整
    imgsz=640,  # 图像大小，YOLO 模型输入的图像尺寸，这里是 640x640
    batch=16,  # 批大小，根据显存和硬件选择合适的值
    device="0",  # 使用的 GPU 设备（如果你有多张 GPU，0 是第一张 GPU，"-1" 表示使用 CPU）
    project="runs/train",  # 训练结果保存路径
    name="exp",  # 训练结果文件夹名称（会在 project 路径下创建一个新文件夹）
    exist_ok=True,  # 如果已经存在同名的训练结果文件夹，允许覆盖
    )

def generate_target_domain_label_txt(train_img_dir, train_label_dir, modelPath):
    # 加载模型（只需要加载一次）
    model = YOLO(modelPath)
    
    # 遍历目标域数据集
    for target_img in os.listdir(train_img_dir):
        if 'snow' in target_img:
            continue
        target_img_path = os.path.join(train_img_dir, target_img)
        target_label_path = os.path.join(train_label_dir, target_img.replace(".jpg", ".txt"))
        if os.path.exists(target_label_path):
            # remove the label file
            os.remove(target_label_path)

        # 预测目标域数据
        results = model.predict(target_img_path)  
        
        # 保存预测结果
        for r in results:  # results是一个列表
            boxes = r.boxes  # 获取预测框
            with open(target_label_path, 'w') as f:
                for box in boxes:
                    # 获取类别和置信度
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    if conf < 0.5:
                        continue
                    # 获取归一化的边界框坐标
                    x, y, w, h = box.xywhn[0].tolist()
                    
                    # 写入YOLO格式
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def dan_tain():
    yamlPath = "model_cfg/yolo11_dan.yaml"
    model = YOLO(yamlPath).load("yolo11n.pt")
    # train and save the model in source domain
    model.train(
        data="datasets/data_source.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="0",
        project="runs/train",
        name="exp",
        exist_ok=True,
    )
    #save the model
    model.save("yolo11n_source.pt")

    #generate the target domain label txt
    generate_target_domain_label_txt("datasets/dawn_dan/train/images", "datasets/dawn_dan/train/labels", "yolo11n_source.pt")

    #train the model in all domain
    model=YOLO(yamlPath).load("yolo11n_source.pt")
    model.train(
        data="datasets/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="0",
        project="runs/train",
        name="exp",
        exist_ok=True,
    )


        
if __name__=='__main__':
    # train("model_cfg/yolo11.yaml", "datasets/data_source.yaml", "yolo11n.pt")
    # dan_tain()

    """
    dawn 数据集: 
    {'2': 3667, '4': 118, '0': 242, '5': 433, '3': 55, '1': 22}
    {'2': 1006, '0': 77, '5': 105, '3': 25, '4': 21, '1': 3}
    """
    # 原始模型
    train("model_cfg/yolo11.yaml", "datasets/data_dawn.yaml")
    # bifpn
    train("model_cfg/yolo11_bifpn.yaml", "datasets/data_dawn.yaml")
    # dcnv4
    train("model_cfg/yolo11_dcnv4.yaml", "datasets/data_dawn.yaml")
    
    
