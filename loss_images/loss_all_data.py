import os
import json
import matplotlib.pyplot as plt
import numpy as np

# 读取日志文件内容并提取 loss 和 iter 信息
def parse_log_files(log_files):
    loss_data = []  # 用于存储所有 epoch 的 loss 数据   
    for log_file in log_files:
        with open(log_file, 'r') as f:
            next(f)  # 跳过第一行
            for line in f:
                    log_data = json.loads(line.strip())
                    if log_data["mode"] == "train":
                        epoch = log_data["epoch"]
                        iter_num = log_data["iter"]
                        loss = log_data["loss"]
                        loss_data.append((epoch, iter_num, loss))  
    loss_all_value = []
    for loss in loss_data:
         loss_all_value.append(loss[2])
    return loss_all_value

#画图函数
def plot_loss_curve(loss_data, save_dir):
    epochs = 20
    iters_per_epoch = 3000  # 每个epoch有3000个iter
    iters = [i for i in range(1, len(loss_data) + 1)]  # 生成iter列表

    # 绘制损失曲线
    plt.figure(figsize=(20, 6))
    plt.plot(iters, loss_data ,label='Loss Curve')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Iterations')
    plt.grid(True)
    plt.legend()

    # 保存图像
    file_path = os.path.join(save_dir, 'loss_all_data_curve.png')
    plt.savefig(file_path, dpi=600)
    plt.close()  # 关闭图像以释放内存



# 日志文件路径列表
log_files = [
    'projects/work_dirs/stage3_e2e_IMU/base_e2e_IMU/20240917_145321.log.json',
    'projects/work_dirs/stage3_e2e_IMU/base_e2e_IMU/20240918_102939.log.json',
    'projects/work_dirs/stage3_e2e_IMU/base_e2e_IMU/20240918_201954.log.json',
    'projects/work_dirs/stage3_e2e_IMU/base_e2e_IMU/20240919_095423.log.json',
    'projects/work_dirs/stage3_e2e_IMU/base_e2e_IMU/20240920_163737.log.json',
    'projects/work_dirs/stage3_e2e_IMU/base_e2e_IMU/20240920_235513.log.json'
]

# 解析日志讲所有loss值记录在list中
loss_data = parse_log_files(log_files)

# 定义保存图片的目录
save_dir = 'projects/work_dirs/stage3_e2e_IMU/base_e2e_IMU/loss_images'
plot_loss_curve(loss_data, save_dir)
