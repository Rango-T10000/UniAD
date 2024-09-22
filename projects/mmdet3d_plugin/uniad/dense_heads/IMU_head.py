import torch
import torch.nn as nn
from mmdet.models.builder import HEADS
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


@HEADS.register_module()
class IMUHead(nn.Module):
    def __init__(self, input_dim=256, traj_dim=2, imu_dim=4, num_heads=8, num_layers=6, mlp_hidden_dim=512):
        super(IMUHead, self).__init__()
        
        # Linear Proj: for sdc_traj, current and previous IMU frame embeddings
        self.traj_proj = nn.Linear(traj_dim, 256)
        self.imu_proj = nn.Linear(imu_dim, 256)
        
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
  
        # Loss function (MSELoss)
        self.criterion = nn.MSELoss()

    def forward_train(self, bev_embed, traj, current_frame_imu, previous_frame_imu, gt_future_frame_e2g_r):
        # 1. Project inputs to the same dimension
        traj_proj = self.traj_proj(traj)  # shape [1, 6, 256]
        current_imu_proj = self.imu_proj(current_frame_imu[0]).unsqueeze(0).unsqueeze(1)# shape [1, 1, 256]
        
        #previous IMU frames处理，根据其实际sample数量进行处理，过去的都堆在一起
        previous_imu_projs = []
        for imu in previous_frame_imu:
            previous_imu_projs.append(self.imu_proj(imu).unsqueeze(0).unsqueeze(1))# shape [1, 1, 256] for each  
        previous_imu_proj = torch.cat(previous_imu_projs, dim=1)  # shape [1, len(previous_frame_imu), 256]
        
        # 2. Concatenate inputs along the time dimension：[1, 1 + len(previous_frame_imu) + 6, 256] 
        input_sequence = torch.cat([current_imu_proj, previous_imu_proj, traj_proj], dim=1)
        
        # 3. 使用Transformer Decoder，将bev_embed作为memory传入
        bev_embed_memory = bev_embed.permute(1, 0, 2)  # 需要将bev_embed的shape转换为 [1, 40000, 256]，[seq_len, batch_size, feature_dim]
        transformer_output = self.transformer_decoder(input_sequence.permute(1, 0, 2), bev_embed_memory)  # shape [total_time_steps, 1, 256]
               
        # 4. reshape transformer输出，传递给MLP预测IMU
        transformer_output = transformer_output.permute(1, 0, 2)  # 回到 [1, total_time_steps, 256]
        total_time_steps = input_sequence.shape[1]
        transformer_output_reshaped = transformer_output[:, total_time_steps - 6:].view(-1, 256)  # reshape to [6, 256]
        imu_predictions = self.prediction_head(transformer_output_reshaped)  # shape [6, 4]
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
        current_imu_proj = self.imu_proj(current_frame_imu[0]).unsqueeze(0).unsqueeze(1)# shape [1, 1, 256]
        
        #previous IMU frames处理，根据其实际sample数量进行处理，过去的都堆在一起
        previous_imu_projs = []
        for imu in previous_frame_imu:
            previous_imu_projs.append(self.imu_proj(imu).unsqueeze(0).unsqueeze(1))# shape [1, 1, 256] for each  
        previous_imu_proj = torch.cat(previous_imu_projs, dim=1)  # shape [1, len(previous_frame_imu), 256]
        
        # 2. Concatenate inputs along the time dimension：[1, 1 + len(previous_frame_imu) + 6, 256] 
        input_sequence = torch.cat([current_imu_proj, previous_imu_proj, traj_proj], dim=1)
        
        # 3. 使用Transformer Decoder，将bev_embed作为memory传入
        bev_embed_memory = bev_embed.permute(1, 0, 2)  # 需要将bev_embed的shape转换为 [1, 40000, 256]，[seq_len, batch_size, feature_dim]
        transformer_output = self.transformer_decoder(input_sequence.permute(1, 0, 2), bev_embed_memory)  # shape [total_time_steps, 1, 256]
               
        # 4. reshape transformer输出，传递给MLP预测IMU
        transformer_output = transformer_output.permute(1, 0, 2)  # 回到 [1, total_time_steps, 256]
        total_time_steps = input_sequence.shape[1]
        transformer_output_reshaped = transformer_output[:, total_time_steps - 6:].view(-1, 256)  # reshape to [6, 256]
        imu_predictions = self.prediction_head(transformer_output_reshaped)  # shape [6, 4]
        imu_predictions = imu_predictions.view(1, 6, 4)  # Reshape back to [1, 6, 4] 

        # 5. 预测误差精度计算
        if len(gt_future_frame_e2g_r) == 0:
            # 如果是空列表，返回一个默认的零精度
            accuracy_value = torch.tensor(0.0).to(imu_predictions.device)  # 确保在正确的设备上计算
            accuracy_IMU = {'accuracy': accuracy_value}
        else:
            gt_future_frame_e2g_r = torch.stack(gt_future_frame_e2g_r, dim=0).unsqueeze(0)  # 一般情况下是shape [1, 6, 4]，有时候例外情况是只有5个sample，即[1,5,4]
            real_future_time_step = gt_future_frame_e2g_r.shape[1]
            # 使用 self.error 函数计算精度
            accuracy_value = self.error(imu_predictions[:, :real_future_time_step, :], gt_future_frame_e2g_r)
            accuracy_IMU = {'accuracy': accuracy_value}
        

        outs_IMU = {
            "predict_future_frame_e2g_r": imu_predictions,
            "accuracy": accuracy_IMU
        }
        
        return outs_IMU

    def loss(self, imu_predictions, gt_future_frame_e2g_r):
        """
        Calculate the MSE loss between predicted IMU data and ground truth IMU data.
        :param imu_predictions: Predicted IMU data of shape [1, 6, 4]
        :param gt_future_frame_e2g_r: Ground truth IMU data of shape [1, 6, 4]
        :return: MSE loss value
        """
        return self.criterion(imu_predictions, gt_future_frame_e2g_r)
    

    def error(self, imu_predictions, gt_future_frame_e2g_r):
        n = imu_predictions.shape[1]  # 获取 n 个 time step
        errors = []
        
        for i in range(n):
            # 获取当前 time step 的预测和 ground truth 的四元数
            pred_quat = imu_predictions[0, i].cpu().numpy()  # 转换为 numpy array
            gt_quat = gt_future_frame_e2g_r[0, i].cpu().numpy()

            # 将四元数转换为欧拉角 (yaw, pitch, roll)
            pred_ypr = Quaternion(pred_quat).yaw_pitch_roll
            gt_ypr = Quaternion(gt_quat).yaw_pitch_roll

            # 将欧拉角从弧度转换为角度
            gt_y, gt_p, gt_r = np.array(gt_ypr) / np.pi * 180
            pred_y, pred_p, pred_r = np.array(pred_ypr) / np.pi * 180

            # 检查并调整角度，如果角度小于 0 则加上 360
            gt_y = gt_y + 360 if gt_y < 0 else gt_y
            gt_p = gt_p + 360 if gt_p < 0 else gt_p
            gt_r = gt_r + 360 if gt_r < 0 else gt_r

            pred_y = pred_y + 360 if pred_y < 0 else pred_y
            pred_p = pred_p + 360 if pred_p < 0 else pred_p
            pred_r = pred_r + 360 if pred_r < 0 else pred_r

            # 计算绝对误差
            abs_error_y = abs(gt_y - pred_y)
            abs_error_p = abs(gt_p - pred_p)
            abs_error_r = abs(gt_r - pred_r)

            # 计算当前 time step 的平均绝对误差
            avg_abs_error = np.mean([abs_error_y, abs_error_p, abs_error_r])
            errors.append(avg_abs_error)

        # 返回 n 个 time step 的误差平均值
        return torch.tensor(np.mean(errors), dtype=torch.float32).to(imu_predictions.device)

