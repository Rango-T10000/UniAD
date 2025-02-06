import torch

# 加载模型权重
checkpoint = torch.load("ckpts/uniad_base_e2e.pth", map_location="cpu")

# 遍历权重并检查数据类型
for name, param in checkpoint['state_dict'].items():
    print(f"{name}: {param.dtype}")
    break  # 查看一个参数的类型即可
