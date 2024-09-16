import torch
import torch.nn as nn
from mmdet.models.builder import HEADS


@HEADS.register_module()
class IMUHead(nn.Module):
    def __init__(self, input_dim=256, traj_dim=2, imu_dim=4, num_heads=8, num_layers=6, mlp_hidden_dim=512, mlp_layers=3):
        super(IMUHead, self).__init__()
        
        # Linear Proj: for bev_embed, sdc_traj, current and previous IMU frame embeddings
        self.bev_embed_proj = nn.Linear(input_dim, 256)
        self.traj_proj = nn.Linear(traj_dim, 256)
        self.imu_proj = nn.Linear(imu_dim, 256)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
        bev_embed_proj = self.bev_embed_proj(bev_embed)  # shape [40000, 1, 256]
        traj_proj = self.traj_proj(traj)  # shape [1, 6, 256]
        current_imu_proj = self.imu_proj(current_frame_imu[0]).unsqueeze(0).unsqueeze(1)# shape [1, 1, 256]
        
        #previous IMU frames处理，根据其实际sample数量进行处理，过去的都堆在一起
        previous_imu_projs = []
        for imu in previous_frame_imu:
            previous_imu_projs.append(self.imu_proj(imu).unsqueeze(0).unsqueeze(1))# shape [1, 1, 256] for each  
        previous_imu_proj = torch.cat(previous_imu_projs, dim=1)  # shape [1, len(previous_frame_imu), 256]
        
        # 2. Concatenate inputs along the time dimension：[1, 1 + len(previous_frame_imu) + 6, 256] 
        input_sequence = torch.cat([current_imu_proj, previous_imu_proj, traj_proj], dim=1)
        
        # 3. Transformer encoder，多所有的输入数据提取特征
        transformer_output = self.transformer_encoder(input_sequence)  # shape [1, total_time_steps, 256]
        
        # 4. 把Transformer encoder给Prediction head来输出关于IMU数据的预测值
        total_time_steps = input_sequence.shape[1]
        transformer_output_reshaped = transformer_output[:, total_time_steps - 6:].view(-1, 256)  # reshape to [6, 256]
        imu_predictions = self.prediction_head(transformer_output_reshaped)  # shape [6, 4]
        imu_predictions = imu_predictions.view(1, 6, 4)# Reshape back to [1, 6, 4] 
        
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

    def forward_test(self, bev_embed, traj, current_frame_imu, previous_frame_imu):
        """
        This function is used for inference and testing.
        It returns only the imu_predictions without calculating the loss.
        """
        # 1. Project inputs to the same dimension
        bev_embed_proj = self.bev_embed_proj(bev_embed)  # shape [40000, 1, 256]
        traj_proj = self.traj_proj(traj)  # shape [1, 6, 256]
        
        # Since current_frame_imu is a list with one element, use current_frame_imu[0]
        current_imu_proj = self.imu_proj(current_frame_imu[0]).unsqueeze(1)  # shape [1, 1, 256]
        
        # Dynamically process previous IMU frames based on their length
        previous_imu_projs = []
        for imu in previous_frame_imu:
            previous_imu_projs.append(self.imu_proj(imu).unsqueeze(1))  # shape [1, 1, 256] for each
        
        # Concatenate all previous IMU projections along the time dimension
        previous_imu_proj = torch.cat(previous_imu_projs, dim=1)  # shape [1, len(previous_frame_imu), 256]
        
        # 2. Concatenate inputs along the time dimension
        input_sequence = torch.cat([current_imu_proj, previous_imu_proj, traj_proj], dim=1)
        # input_sequence shape: [1, 1 + len(previous_frame_imu) + 6, 256] 
        
        # 3. Transformer encoder
        transformer_output = self.transformer_encoder(input_sequence)  # shape [1, total_time_steps, 256]
        
        # 4. Use the transformer output to predict future IMU data
        total_time_steps = input_sequence.shape[1]
        imu_predictions = self.prediction_head(transformer_output[:, total_time_steps - 6:])  # shape [1, 6, 4] (future 6 steps)

        outs_IMU = {
            "predict_future_frame_e2g_r": imu_predictions,
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
