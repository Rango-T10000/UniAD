import os
import json
import matplotlib.pyplot as plt

# 读取日志文件内容并提取 loss 和 epoch 信息
def parse_log_files(log_files):
    epoch_loss_dict = {}  #出事话一个epoch-loss对应空字典
    #---------遍历传入的列表中每一个log.json文件----------
    for log_file in log_files:
        with open(log_file, 'r') as f: 
            next(f)  # 跳过文件的第一行
            for line in f: #逐行读取
                log_data = json.loads(line.strip())
                if log_data["mode"] == "train":
                    epoch = log_data["epoch"]
                    loss = log_data["loss"]
                    #先检查当前的epoch号是否已经存在于字典中，不存在就添加
                    if epoch not in epoch_loss_dict:
                        epoch_loss_dict[epoch] = []                   
                    epoch_loss_dict[epoch].append(loss)
    
    # 计算每个epoch的平均loss
    avg_epoch_loss = {epoch: sum(losses) / len(losses) for epoch, losses in epoch_loss_dict.items()}
    return avg_epoch_loss

# 绘制loss曲线并保存为图片
def plot_loss_curve(epoch_loss_dict, save_dir):
    epochs = sorted(epoch_loss_dict.keys())
    avg_losses = [epoch_loss_dict[epoch] for epoch in epochs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_losses, marker='o', label="Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Average Training Loss over Epochs")
    plt.grid(True)
    plt.legend()

    # 确保保存路径存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存图片到指定目录，设置dpi提高分辨率
    save_path = os.path.join(save_dir, 'loss_epoch_curve.png')
    plt.savefig(save_path, dpi=600)  # 设置dpi为300
    plt.close()  # 关闭图像以节省内存


#-----------------------------------开始配置作图信息------------------------------------
# 日志文件路径列表
log_files = ['projects/work_dirs/stage3_e2e_IMU/base_e2e_IMU/20240930_100742.log.json']

# 调用解析日志函数解析log.json文件
epoch_loss_dict = parse_log_files(log_files)
# 定义保存图片的目录
save_dir = 'projects/work_dirs/stage3_e2e_IMU/base_e2e_IMU/loss_images'
#--------调用该函数作图-------
plot_loss_curve(epoch_loss_dict, save_dir)
