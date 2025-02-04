from ultralytics import YOLO  # 导入 YOLO 模型


def train(yamlPath, dataPath, modelName="yolo11n.pt"):
    # 1. 加载预训练模型
    model = YOLO(yamlPath).load(modelName)  # 这里假设你已经下载并使用了一个预训练模型的路径
    # model = YOLO("yolo11n.pt")
    # 2. 训练模型
    train_results = model.train(
    data=dataPath,  # 数据集配置文件路径，这里你应该提供正确的 YAML 配置文件路径
    epochs=100,  # 训练的轮数，可以根据需要调整
    imgsz=640,  # 图像大小，YOLO 模型输入的图像尺寸，这里是 640x640
    batch=32,  # 批大小，根据显存和硬件选择合适的值
    device="0",  # 使用的 GPU 设备（如果你有多张 GPU，0 是第一张 GPU，"-1" 表示使用 CPU）
    project="runs/train",  # 训练结果保存路径
    name="exp",  # 训练结果文件夹名称（会在 project 路径下创建一个新文件夹）
    exist_ok=True,  # 如果已经存在同名的训练结果文件夹，允许覆盖
    amp=False
    )

if __name__=='__main__':
    
    """
    3090 ;100 epochs; 640*640; batch=32;
    dawn数据集:
    {2: 5186, 4: 137, 0: 369, 5: 525, 3: 55, 1: 23}
    {2: 1268, 0: 108, 5: 121, 3: 26, 4: 24, 1: 3}

    bdd10k数据集:
    {2: 66263, 1: 22368, 3: 8789, 0: 17788, 4: 1123, 7: 694, 5: 2677, 6: 457, 8: 319, 10: 82, 11: 31, 9: 13, 12: 5}
    {2: 9798, 1: 2488, 0: 3345, 5: 1154, 6: 143, 7: 436, 3: 49, 4: 46, 8: 56, 9: 6, 10: 1}
    
    bdd100k数据集:
    {0: 34724, 1: 26884, 2: 102837, 3: 658, 4: 460, 5: 13425, 6: 1660, 7: 4243, 8: 1039, 9: 85, 10: 15, 11: 2, 12: 1}
    """
    # 原始模型
    train("model_cfg/yolo11_all.yaml", "datasets/data_dawn.yaml")
    # bifpn
    train("model_cfg/yolo11_bifpn.yaml", "datasets/data_dawn.yaml")
    # dcnv4
    train("model_cfg/yolo11_dcnv4.yaml", "datasets/data_dawn.yaml")
    # c2dan
    train("model_cfg/yolo11_c2dan.yaml", "datasets/data_dawn.yaml")
