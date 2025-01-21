import torch
import torch.nn as nn
from mmdet.models.builder import HEADS
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import math


#------------------------------transformer—decoder-----------------------------------
class AddNorm(nn.Module):
    def __init__(self, embedding_dim):
        super(AddNorm, self).__init__()
        self.add_norm = nn.LayerNorm(embedding_dim)

    def forward(self, X, X1):
        X = X + X1
        X = self.add_norm(X)
        return X

class Pos_FFN(nn.Module):
    def __init__(self, embedding_dim, ffn_hidden_dim):
        super(Pos_FFN, self).__init__()
        self.lin1 = nn.Linear(embedding_dim, ffn_hidden_dim)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(ffn_hidden_dim, embedding_dim)

    def forward(self, X):
        X = self.lin1(X)
        X = self.relu1(X)
        X = self.lin2(X)
        return X

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)  #我的任务这里不需要mask
        self.add_norm1 = AddNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.cross_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.add_norm2 = AddNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ffn = Pos_FFN(embedding_dim, ffn_hidden_dim)
        self.add_norm3 = AddNorm(embedding_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory): #memory:the output of the encoder
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt2 = self.dropout1(tgt2)
        tgt = self.add_norm1(tgt, tgt2)
        
        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt2 = self.dropout2(tgt2)
        tgt = self.add_norm2(tgt, tgt2)
        
        tgt2 = self.ffn(tgt)
        tgt2 = self.dropout3(tgt2)
        tgt = self.add_norm3(tgt, tgt2)
        
        return tgt

class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_hidden_dim, num_layers, output_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(embedding_dim, num_heads, ffn_hidden_dim, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        tgt = self.linear(tgt)
        return tgt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)


#------------------------------model define-----------------------------------
@HEADS.register_module()
class IMU_head(nn.Module):
    def __init__(self, traj_dim=2, imu_dim=4, max_len=6, embedding_dim=256, num_heads=8, ffn_hidden_dim=1024, num_layers=6, output_dim=256, mlp_hidden_dim=512, device='cuda'):
        super(IMU_head, self).__init__()
        
        # Linear Proj: for sdc_traj embeddings
        self.traj_proj = nn.Linear(traj_dim, embedding_dim) # nn.Linear(2,256)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len, device)
        
        # Transformer Decoder 
        self.decoder = Decoder(embedding_dim, num_heads, ffn_hidden_dim, num_layers, output_dim)
        
        #MLP as prediction head
        mlp_layers_list = []
        # 第一层：输入维度是 256，输出维度是 512
        mlp_layers_list.append(nn.Linear(output_dim, mlp_hidden_dim)) 
        mlp_layers_list.append(nn.ReLU())
        mlp_layers_list.append(nn.LayerNorm(mlp_hidden_dim))  
        # 第二层：输入和输出都是 512
        mlp_layers_list.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))  
        mlp_layers_list.append(nn.ReLU())
        mlp_layers_list.append(nn.LayerNorm(mlp_hidden_dim)) 
        mlp_layers_list.append(nn.Linear(mlp_hidden_dim, imu_dim))  
        self.prediction_head = nn.Sequential(*mlp_layers_list)

    def forward_train(self, bev_embed, traj, current_frame_imu, previous_frame_imu, gt_future_frame_e2g_r):
        # 1. Project inputs to the same dimension
        traj_proj = self.traj_proj(traj)  # (1,6,2)--> (1,6,256)
        
        # 2. 添加位置编码
        traj_proj = traj_proj + self.positional_encoding(traj_proj)

        # 3. 使用Transformer Decoder，将bev_embed作为memory传入
        traj_proj = traj_proj.permute(1, 0, 2)  # (1,6,256) -> (6,1,256):这是nn.MultiheadAttention的输入要求; bev_embed本来就是(40000,1,256)不用转换
        decoder_output = self.decoder(traj_proj, bev_embed)
        decoder_output = decoder_output.permute(1, 0, 2)  # (sequence_length, batch_size, embedding_dim) -> (batch_size, sequence_length, embedding_dim)
        
        # 4. 传递给MLP预测IMU
        imu_predictions = self.prediction_head(decoder_output)
        
        # 5. 将预测值转化为单位四元数
        imu_predictions_norm = torch.norm(imu_predictions, dim=-1, keepdim=True)  # 计算四元数的范数
        imu_predictions = imu_predictions / imu_predictions_norm  # 归一化四元数，使其范数为1        

        # 6. 预测误差精度计算
        if len(gt_future_frame_e2g_r) == 0:
            # 如果是空列表，返回一个默认的零精度
            loss_value = torch.tensor(0.0, device = imu_predictions.device , requires_grad=True)  # 确保在正确的设备上计算
            losses_IMU = {'losses': loss_value,}
        else:
            gt_future_frame_e2g_r = torch.stack(gt_future_frame_e2g_r, dim=0).unsqueeze(0)
            real_future_time_step = gt_future_frame_e2g_r.shape[1]
            # 使用 self.loss 函数计算loss
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
        traj_proj = self.traj_proj(traj)  # (1,6,2)--> (1,6,256)
        
        # 2. 添加位置编码
        traj_proj = traj_proj + self.positional_encoding(traj_proj)
        
        # 3. 使用Transformer Decoder，将bev_embed作为memory传入
        traj_proj = traj_proj.permute(1, 0, 2)  # (1,6,256) -> (6,1,256):这是nn.MultiheadAttention的输入要求; bev_embed本来就是(40000,1,256)不用转换
        decoder_output = self.decoder(traj_proj, bev_embed)
        decoder_output = decoder_output.permute(1, 0, 2)  # (sequence_length, batch_size, embedding_dim) -> (batch_size, sequence_length, embedding_dim)
        
        # 4. 传递给MLP预测IMU
        imu_predictions = self.prediction_head(decoder_output)
        
        # 5. 将预测值转化为单位四元数
        imu_predictions_norm = torch.norm(imu_predictions, dim=-1, keepdim=True)  # 计算四元数的范数
        imu_predictions = imu_predictions / imu_predictions_norm  # 归一化四元数，使其范数为1        

        # 6. 预测误差精度计算
        if len(gt_future_frame_e2g_r) == 0:
            # 如果是空列表，返回一个默认的零精度
            accuracy_value = torch.tensor(0.0).to(imu_predictions.device)  # 确保在正确的设备上计算
            accuracy_IMU = {'accuracy': accuracy_value}
        else:
            gt_future_frame_e2g_r = torch.stack(gt_future_frame_e2g_r, dim=0).unsqueeze(0)
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
        # 初始化损失和计数器
        total_loss = 0.0
        count = 0

        # 遍历每个时间步
        for t in range(imu_predictions.shape[1]):
            pred = imu_predictions[0, t, :]
            gt = gt_future_frame_e2g_r[0, t, :]

            if not torch.isnan(gt).any():
                # 计算四元数的点积
                dot_product = torch.sum(pred * gt, dim=-1)
                # 确保 cos_theta 在 [-1, 1] 之间
                cos_theta = torch.clamp(dot_product, -1.0, 1.0)
                # 计算损失
                loss = 1.0 - torch.abs(cos_theta)
                total_loss = total_loss + loss
                count = count + 1


        # # 返回所有时间步的平均损失
        # return total_loss / count if count > 0 else torch.tensor(0.0, device = imu_predictions.device , requires_grad=True)
        if count > 0:
            loss_value = total_loss / count
            return loss_value
        else:
            loss_value = torch.tensor(0.0, device = imu_predictions.device , requires_grad=True)
            return loss_value
    
    def error(self, imu_predictions, gt_future_frame_e2g_r):
        # 初始化损失和计数器
        total_loss = 0.0
        count = 0

        # 遍历每个时间步
        for t in range(imu_predictions.shape[1]):
            pred = imu_predictions[0, t, :]
            gt = gt_future_frame_e2g_r[0, t, :]
            if not torch.isnan(gt).any():
                # 计算四元数的点积
                dot_product = torch.sum(pred * gt, dim=-1)
                # 确保 cos_theta 在 [-1, 1] 之间
                cos_theta = torch.clamp(dot_product, -1.0, 1.0)
                # 计算损失
                loss = 1.0 - torch.abs(cos_theta)
                total_loss += loss
                count += 1


        # # 返回所有时间步的平均损失
        # return total_loss / count if count > 0 else torch.tensor(0.0).to(imu_predictions.device)
        if count > 0:
            loss_value = total_loss / count
            return loss_value
        else:
            loss_value = torch.tensor(0.0).to(imu_predictions.device)
            return loss_value