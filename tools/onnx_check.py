import onnx
import numpy as np

# 加载导出的 ONNX 模型
onnx_model = onnx.load(r"D:\1_SHU_programming\SRM\ultralytics-main\datasets\runs\detect\train4\weights\best.onnx")

# 遍历 ONNX 模型的所有节点，检查是否有 INT64 类型的张量
for tensor in onnx_model.graph.initializer:
    if tensor.data_type == onnx.TensorProto.INT64:
        print(f"Tensor {tensor.name} is INT64, converting to INT32.")
        # 将 INT64 转换为 INT32
        int64_data = np.frombuffer(tensor.raw_data, dtype=np.int64)
        int32_data = int64_data.astype(np.int32)
        tensor.raw_data = int32_data.tobytes()
        tensor.data_type = onnx.TensorProto.INT32

# 保存修改后的 ONNX 模型
onnx.save(onnx_model, "runs/detect/train4/weights/best_int32.onnx")
