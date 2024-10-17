import torch
from ultralytics import YOLO

# 加载训练后的 YOLO 模型
model = YOLO(r"D:\1_SHU_programming\SRM\ultralytics-main\datasets\runs\detect\train\weights\best.pt")

# 检查并将模型中的 int64 类型参数转换为 int32
for param in model.model.parameters():
    if param.dtype == torch.int64:  # 检查是否为 int64
        param.data = param.data.to(torch.int32)  # 转换为 int32

# 将模型量化为 INT8 格式并导出为 ONNX
imgsz = (640, 640)

# 如果你使用的是量化后的模型，可以使用类似下面的代码：
model.export(
    format='onnx',          # 导出为 ONNX 格式
    imgsz=imgsz,            # 图像尺寸
    int8=True,              # 保留 INT8 量化
    half=False,             # 禁用半精度（FP16），以防模型在推理中出现浮点数错误
    dynamic=False,           # 动态批量大小支持
    simplify=True           # 简化 ONNX 模型，移除不必要的节点
)

print("模型导出成功，已转换为 ONNX 格式")
