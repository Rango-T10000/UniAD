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
    return loss_data

# 计算每500个iter的平均loss
def calculate_average_losses(loss_data):
    average_losses = []
    
    # 每个 epoch 有 3000 个 iter，分成 6 个部分，每个部分 500
    for epoch_index in range(20):  # 假设有 20 个 epochs
        # 获取当前 epoch 的数据
        epoch_data = [item for item in loss_data if item[0] == epoch_index + 1]
        
        # 每 500 个 iter 计算平均值
        epoch_avg_losses = []
        for i in range(0, len(epoch_data), 50):  # 每个 epoch 有 60 个 500 的部分
            batch = epoch_data[i:i + 50]
            if not batch:
                continue
            avg_loss = sum(item[2] for item in batch) / len(batch)
            epoch_avg_losses.append(avg_loss)
        
        average_losses.append(epoch_avg_losses)
        iter_500_loss = [loss for epoch_losses in average_losses for loss in epoch_losses]
    
    return iter_500_loss

#画图函数
def plot_loss_curve(iter_500_loss, save_dir):
    epochs = 20
    iters_per_epoch = 3000  # 每个epoch有3000个iter
    iters = [i * 500 for i in range(1, len(iter_500_loss) + 1)]  # 生成iter列表


    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(iters, iter_500_loss, marker='o' ,label='Loss Curve')

    # 设置横轴刻度
    epoch_ticks = [i * iters_per_epoch for i in range(1, epochs + 1)]
    epoch_ticks.insert(0,0)
    plt.xticks(epoch_ticks, range(0, epochs + 1))  # 标出epoch

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Iterations')
    plt.grid(True)
    plt.legend()

    # 保存图像
    file_path = os.path.join(save_dir, 'loss_iter_curve.png')
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

# 解析日志并计算平均loss
loss_data = parse_log_files(log_files)
iter_500_loss = calculate_average_losses(loss_data)

# 定义保存图片的目录
save_dir = 'projects/work_dirs/stage3_e2e_IMU/base_e2e_IMU/loss_images'
plot_loss_curve(iter_500_loss, save_dir)
