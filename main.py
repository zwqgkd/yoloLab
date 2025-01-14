from ultralytics import YOLO  # 导入 YOLO 模型


def train(yamlPath, dataPath, modelName="yolo11s.pt"):
    # 1. 加载预训练模型
    model = YOLO(yamlPath).load(modelName)  # 这里假设你已经下载并使用了一个预训练模型的路径
    # model = YOLO("yolo11n.pt")
    # 2. 训练模型
    train_results = model.train(
    data=dataPath,  # 数据集配置文件路径，这里你应该提供正确的 YAML 配置文件路径
    epochs=100,  # 训练的轮数，可以根据需要调整
    imgsz=640,  # 图像大小，YOLO 模型输入的图像尺寸，这里是 640x640
    batch=4,  # 批大小，根据显存和硬件选择合适的值
    device="0",  # 使用的 GPU 设备（如果你有多张 GPU，0 是第一张 GPU，"-1" 表示使用 CPU）
    project="runs/train",  # 训练结果保存路径
    name="exp",  # 训练结果文件夹名称（会在 project 路径下创建一个新文件夹）
    exist_ok=True,  # 如果已经存在同名的训练结果文件夹，允许覆盖
    amp=False
    )

if __name__=='__main__':
    # train("model_cfg/yolo11.yaml", "datasets/data.yaml", "yolo11n.pt") #543 345
    train("model_cfg/yolo11_dan.yaml", "datasets/data.yaml", "yolo11n.pt") #543 345
    # train("model_cfg/yolo11_bifpn.yaml", "datasets/data.yaml", "yolo11n.pt") #0.
