import torch
import torch.nn as nn
from mmdet.models.builder import HEADS
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import math


@HEADS.register_module()
class IMUHead(nn.Module):
    def __init__(self, input_dim=256, traj_dim=2, imu_dim=4, num_heads=8, num_layers=6, mlp_hidden_dim=512, max_len=6):
        super(IMUHead, self).__init__()
        
        # Linear Proj: for sdc_traj, current and previous IMU frame embeddings
        self.traj_proj = nn.Linear(traj_dim, 256)
        # self.imu_proj = nn.Linear(imu_dim, 256)

        # 可学习的位置编码，维度为 [max_len, 2]
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, input_dim))
        
        # Transformer Decoder (using bev_embed as K，V and input_sequence as query)
        decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # MLP for prediction
        mlp_layers_list = []
        # 第一层：输入维度是 256，输出维度是 512
        mlp_layers_list.append(nn.Linear(256, mlp_hidden_dim))  # input 256, output 512
        mlp_layers_list.append(nn.ReLU())
        mlp_layers_list.append(nn.LayerNorm(mlp_hidden_dim))  # Layer normalization

        # 第二层：输入和输出都是 512
        mlp_layers_list.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))  # input 512, output 512
        mlp_layers_list.append(nn.ReLU())
        mlp_layers_list.append(nn.LayerNorm(mlp_hidden_dim))  # Layer normalization

        # 最后一层：输入是 512，输出是 IMU 的维度 4
        mlp_layers_list.append(nn.Linear(mlp_hidden_dim, imu_dim))  # Final layer for predicting IMU

        self.prediction_head = nn.Sequential(*mlp_layers_list)

    def forward_train(self, bev_embed, traj, current_frame_imu, previous_frame_imu, gt_future_frame_e2g_r):
        # 1. Project inputs to the same dimension
        traj_proj = self.traj_proj(traj)  # shape [1, 6, 256]
        traj_len = traj_proj.shape[1]
        traj_proj = traj_proj + self.positional_encoding[:, :traj_len, :]  # 添加位置编码
        # current_imu_proj = self.imu_proj(current_frame_imu[0]).unsqueeze(0).unsqueeze(1)# shape [1, 1, 256]
        
        # #previous IMU frames处理，根据其实际sample数量进行处理，过去的都堆在一起
        # previous_imu_projs = []
        # for imu in previous_frame_imu:
        #     previous_imu_projs.append(self.imu_proj(imu).unsqueeze(0).unsqueeze(1))# shape [1, 1, 256] for each  
        # previous_imu_proj = torch.cat(previous_imu_projs, dim=1)  # shape [1, len(previous_frame_imu), 256]
        
        # # 2. Concatenate inputs along the time dimension：[1, 1 + len(previous_frame_imu) + 6, 256] 
        # input_sequence = torch.cat([current_imu_proj, previous_imu_proj, traj_proj], dim=1)
        input_sequence = traj_proj

        # 3. 使用Transformer Decoder，将bev_embed作为memory传入
        bev_embed_memory = bev_embed.permute(1, 0, 2)  # 需要将bev_embed的shape转换为 [1, 40000, 256]，[seq_len, batch_size, feature_dim]
        # transformer_output = self.transformer_decoder(input_sequence.permute(1, 0, 2), bev_embed_memory)  # shape [total_time_steps, 1, 256]
        transformer_output = self.transformer_decoder(input_sequence.permute(1, 0, 2), bev_embed_memory)
               
        # 4. reshape transformer输出，传递给MLP预测IMU
        # transformer_output = transformer_output.permute(1, 0, 2)  # 回到 [1, total_time_steps, 256]
        # total_time_steps = input_sequence.shape[1]
        # transformer_output_reshaped = transformer_output[:, total_time_steps - 6:].view(-1, 256)  # reshape to [6, 256]
        # imu_predictions = self.prediction_head(transformer_output_reshaped)  # shape [6, 4]
        imu_predictions = self.prediction_head(transformer_output) 
        
        #将预测值转化为单位四元数
        imu_predictions_norm = torch.norm(imu_predictions, dim=-1, keepdim=True)  # 计算四元数的范数
        imu_predictions = imu_predictions / imu_predictions_norm  # 归一化四元数，使其范数为1        
        imu_predictions = imu_predictions.view(1, 6, 4)  # Reshape back to [1, 6, 4] 

        # 5. Compute the loss using the predicted and ground truth IMU data
        if len(gt_future_frame_e2g_r) == 0:
            # 如果是空列表，返回一个默认的零损失
            loss_value = torch.tensor(0.0, requires_grad=True).to(imu_predictions.device)  # 确保在正确的设备上计算
            losses_IMU = {'losses': loss_value}
        else:
            gt_future_frame_e2g_r = torch.stack(gt_future_frame_e2g_r, dim=0).unsqueeze(0)  # 一般情况下是shape [1, 6, 4]，有时候例外情况是只有5个sample，即[1,5,4]
            real_future_time_step = gt_future_frame_e2g_r.shape[1]
            loss_value = self.loss(imu_predictions[:, :real_future_time_step, :], gt_future_frame_e2g_r)
            losses_IMU = {'losses': loss_value,}

        outs_IMU = {
            "predict_future_frame_e2g_r": imu_predictions,
            "losses": losses_IMU
        }
        
        return outs_IMU

    def forward_test(self, bev_embed, traj, current_frame_imu, previous_frame_imu, gt_future_frame_e2g_r):
        """
        This function is used for inference and testing.
        It returns only the imu_predictions without calculating the loss.
        """
        # 1. Project inputs to the same dimension
        traj_proj = self.traj_proj(traj)  # shape [1, 6, 256]
        traj_len = traj_proj.shape[1]
        traj_proj = traj_proj + self.positional_encoding[:, :traj_len, :]  # 添加位置编码
        # current_imu_proj = self.imu_proj(current_frame_imu[0]).unsqueeze(0).unsqueeze(1)# shape [1, 1, 256]
        
        # #previous IMU frames处理，根据其实际sample数量进行处理，过去的都堆在一起
        # previous_imu_projs = []
        # for imu in previous_frame_imu:
        #     previous_imu_projs.append(self.imu_proj(imu).unsqueeze(0).unsqueeze(1))# shape [1, 1, 256] for each  
        # previous_imu_proj = torch.cat(previous_imu_projs, dim=1)  # shape [1, len(previous_frame_imu), 256]
        
        # # 2. Concatenate inputs along the time dimension：[1, 1 + len(previous_frame_imu) + 6, 256] 
        # input_sequence = torch.cat([current_imu_proj, previous_imu_proj, traj_proj], dim=1)
        input_sequence = traj_proj

        # 3. 使用Transformer Decoder，将bev_embed作为memory传入
        bev_embed_memory = bev_embed.permute(1, 0, 2)  # 需要将bev_embed的shape转换为 [1, 40000, 256]，[seq_len, batch_size, feature_dim]
        # transformer_output = self.transformer_decoder(input_sequence.permute(1, 0, 2), bev_embed_memory)  # shape [total_time_steps, 1, 256]
        transformer_output = self.transformer_decoder(input_sequence.permute(1, 0, 2), bev_embed_memory)
               
        # 4. reshape transformer输出，传递给MLP预测IMU
        # transformer_output = transformer_output.permute(1, 0, 2)  # 回到 [1, total_time_steps, 256]
        # total_time_steps = input_sequence.shape[1]
        # transformer_output_reshaped = transformer_output[:, total_time_steps - 6:].view(-1, 256)  # reshape to [6, 256]
        # imu_predictions = self.prediction_head(transformer_output_reshaped)  # shape [6, 4]
        imu_predictions = self.prediction_head(transformer_output) 
        
        #将预测值转化为单位四元数
        imu_predictions_norm = torch.norm(imu_predictions, dim=-1, keepdim=True)  # 计算四元数的范数
        imu_predictions = imu_predictions / imu_predictions_norm  # 归一化四元数，使其范数为1        
        imu_predictions = imu_predictions.view(1, 6, 4)  # Reshape back to [1, 6, 4] 

        # 5. 预测误差精度计算
        if len(gt_future_frame_e2g_r) == 0:
            # 如果是空列表，返回一个默认的零精度
            accuracy_value = torch.tensor(0.0).to(imu_predictions.device)  # 确保在正确的设备上计算
            accuracy_IMU = {'accuracy': accuracy_value}
        else:
            gt_future_frame_e2g_r = torch.stack(gt_future_frame_e2g_r, dim=0).permute(1, 0, 2)  # 一般情况下是shape [1, 6, 4]，有时候例外情况是只有5个sample，即[1,5,4]
            real_future_time_step = gt_future_frame_e2g_r.shape[1]
            # 使用 self.error 函数计算精度
            accuracy_value = self.error(imu_predictions[:, :real_future_time_step, :], gt_future_frame_e2g_r)
            accuracy_IMU = {'accuracy': accuracy_value}
        

        outs_IMU = {
            "predict_future_frame_e2g_r": imu_predictions,
            "accuracy": accuracy_IMU
        }

        return outs_IMU

    def quaternion_to_euler(self, q):
        """将四元数转换为欧拉角 (yaw, pitch, roll)"""

        #先判断是否为单位四元数
        sum_of_squares = torch.dot(q, q)
        if abs(1.0 - sum_of_squares) > 1e-14:
            norm = torch.sqrt(sum_of_squares)
            if norm > 0:
                q = q /norm

        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        #以下公式应该要确定是单位四元数吧，再检查公式
        # Calculate yaw (around z-axis)
        yaw = torch.atan2(2 * (w * z - x * y), 1 - 2 * (y**2 + z**2))

        # Calculate pitch (around y'-axis)
        pitch = torch.asin(2 * (w * y + z * x))

        # Calculate roll (around x''-axis)
        roll = torch.atan2(2 * (w * x - y * z), 1 - 2 * (x**2 + y**2))

        return torch.stack([yaw, pitch, roll], dim=-1)  # 返回 [yaw, pitch, roll]

    def loss(self, imu_predictions, gt_future_frame_e2g_r):
        """
        Calculate the loss between predicted IMU data and ground truth IMU data.
        :param imu_predictions: Predicted IMU data of shape [1, 6, 4]
        :param gt_future_frame_e2g_r: Ground truth IMU data of shape [1, 6, 4]
        """
        # return self.criterion(imu_predictions, gt_future_frame_e2g_r)
        #MSE作为loss并不合理，这忽略了四元数的几何意义，我是希望两个Quaternion所表示的角度尽可能一致，而不是Quaternion数值本身
        #思路1: 先把Quaternion转换为欧拉角，并规范到[0~360 degree]，再计算三个角度的mse作为loss
        #思路2: 计算两个Quaternion的点积，再换算为角度，把这个角度作为loss
        
        #------------1-------------
        #对每个时间步的预测和真实值计算四元数的点积,因为传进来的就是单位四元数，不用再归一化
        dot_product = torch.sum(imu_predictions * gt_future_frame_e2g_r, dim=-1)
        # 分别计算预测值和真实值的模长
        # pred_norm = torch.norm(imu_predictions, dim=-1)  # shape: [1, n]
        # gt_norm = torch.norm(gt_future_frame_e2g_r, dim=-1)  # shape: [1, n]
        cos_theta = torch.clamp(dot_product, -1.0, 1.0) #保证得到的cos_theta在[-1,1]
        loss = 1.0 - torch.abs(cos_theta)
        # 返回所有时间步的平均损失
        return torch.mean(loss)

        #------------2-------------
        # # 对每个时间步的预测和真实值计算四元数的点积
        # dot_product = torch.sum(imu_predictions * gt_future_frame_e2g_r, dim=-1)
        # dot_product = torch.clamp(dot_product, -1.0, 1.0)  # 确保点积的值在合法范围内
        # angle_diff = 2 * torch.acos(torch.abs(dot_product))  # 计算角度差(单位是rad)
        
        # # 返回平均角度差作为 loss
        # return torch.mean(angle_diff)
    
        #------------3------------
        # n = imu_predictions.shape[1]  # 获取 n 个 time step
        # errors = []

        # for i in range(n):
        #     # 获取当前 time step 的预测和 ground truth 的四元数
        #     pred_quat = imu_predictions[0, i]  # 直接保留为 Tensor
        #     gt_quat = gt_future_frame_e2g_r[0, i]

        #     # 将四元数转换为欧拉角 (yaw, pitch, roll)
        #     pred_ypr = self.quaternion_to_euler(pred_quat)  # 预测欧拉角
        #     gt_ypr = self.quaternion_to_euler(gt_quat)  # Ground truth 欧拉角

        #     # 将欧拉角从弧度转换为角度
        #     pi_tensor = torch.tensor(np.pi, device=imu_predictions.device)
        #     gt_ypr = gt_ypr / pi_tensor * 180
        #     pred_ypr = pred_ypr / pi_tensor * 180

        #     # 调整角度，如果角度小于 0 则加上 360
        #     gt_ypr = torch.where(gt_ypr < 0, gt_ypr + 360, gt_ypr)
        #     pred_ypr = torch.where(pred_ypr < 0, pred_ypr + 360, pred_ypr)

        #     # 计算绝对误差
        #     error_angle = torch.abs(gt_ypr - pred_ypr)
        #     abs_error = torch.where(error_angle > 180, 360 - error_angle, error_angle)

        #     # 计算当前 time step 的平均绝对误差
        #     avg_abs_error = torch.mean(abs_error)
        #     errors.append(avg_abs_error)

        # # 返回 n 个 time step 的误差平均值
        # return torch.mean(torch.stack(errors))

    def error(self, imu_predictions, gt_future_frame_e2g_r):

        # 计算每个时间步的预测和真实值的四元数点积
        dot_product = torch.sum(imu_predictions * gt_future_frame_e2g_r, dim=-1)
        # 确保 cos_theta 在 [-1, 1] 之间
        cos_theta = torch.clamp(dot_product, -1.0, 1.0)
        # 计算损失
        loss = 1.0 - torch.abs(cos_theta)
        # 返回所有时间步的平均损失
        return torch.mean(loss)
    

        # n = imu_predictions.shape[1]  # 获取 n 个 time step
        # errors = []
        
        # for i in range(n):
        #     # 获取当前 time step 的预测和 ground truth 的四元数
        #     pred_quat = imu_predictions[0, i].cpu().numpy()  # 转换为 numpy array
        #     gt_quat = gt_future_frame_e2g_r[0, i].cpu().numpy()

        #     # 将四元数转换为欧拉角 (yaw, pitch, roll)
        #     pred_ypr = Quaternion(pred_quat).yaw_pitch_roll
        #     gt_ypr = Quaternion(gt_quat).yaw_pitch_roll

        #     # 将欧拉角从弧度转换为角度
        #     gt_y, gt_p, gt_r = np.array(gt_ypr) / np.pi * 180
        #     pred_y, pred_p, pred_r = np.array(pred_ypr) / np.pi * 180

        #     # 检查并调整角度，如果角度小于 0 则加上 360
        #     gt_y = gt_y + 360 if gt_y < 0 else gt_y
        #     gt_p = gt_p + 360 if gt_p < 0 else gt_p
        #     gt_r = gt_r + 360 if gt_r < 0 else gt_r

        #     pred_y = pred_y + 360 if pred_y < 0 else pred_y
        #     pred_p = pred_p + 360 if pred_p < 0 else pred_p
        #     pred_r = pred_r + 360 if pred_r < 0 else pred_r

        #     # 计算绝对误差
        #     abs_error_y = 360 - abs(gt_y - pred_y) if abs(gt_y - pred_y) > 180 else abs(gt_y - pred_y)
        #     abs_error_p = 360 - abs(gt_p - pred_p) if abs(gt_p - pred_p) > 180 else abs(gt_p - pred_p)
        #     abs_error_r = 360 - abs(gt_r - pred_r) if abs(gt_r - pred_r) > 180 else abs(gt_r - pred_r)

        #     # 计算当前 time step 的平均绝对误差
        #     avg_abs_error = np.mean([abs_error_y, abs_error_p, abs_error_r])
        #     errors.append(avg_abs_error)

        # # 返回 n 个 time step 的误差平均值
        # return torch.tensor(np.mean(errors), dtype=torch.float32).to(imu_predictions.device)

