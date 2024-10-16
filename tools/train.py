from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-mlca.yaml')  # build a new model from YAML
#model = YOLO('../../yolov8n.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n-mlca.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

if __name__ == '__main__':

    model.train(
        data=r'D:\1_SHU_programming\SRM\ultralytics-main\datasets\data.yaml',  # 训练数据集
        epochs=100,  # 训练的轮数
        imgsz=640,  # 输入图像的大小
        batch=16,  # 每批的图像数量
        # device='',  # 运行的设备，例如 cuda:0 或 cpu
        # workers=8,  # 数据加载的工作线程数
        # project='runs/train',  # 项目名称（可选）
        # name='exp',  # 实验名称，结果保存在'project/name'目录下（可选）
        save_period=5,  # 每5个epoch保存一次模型
        # sync_bn=False,  # 是否使用同步批量归一化
        # freeze=[0],  # 冻结的层（0表示冻结所有层）
        # hyp='hyp.yaml',  # 超参数文件路径
    )