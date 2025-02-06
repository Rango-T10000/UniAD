import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

#下面两个数值一样，符号相反的四元数所表示的角度一模一样！
#注意用四元数举例子不用胡乱举例，
gt = [ 0.9693,  0.0110,  0.0021, -0.2455]
pred = [ 0.6566, -0.5746,  0.1109,  0.3090]

# 获取 gt 和 pred 的欧拉角
gt_ypr = Quaternion(gt).yaw_pitch_roll
pred_ypr = Quaternion(pred).yaw_pitch_roll

# 将每个角度从弧度转换为角度
gt_y, gt_p, gt_r = np.array(gt_ypr) / np.pi * 180
pred_y, pred_p, pred_r = np.array(pred_ypr) / np.pi * 180

# 检查并调整角度，如果角度小于 0 则加上 360
gt_y = gt_y + 360 if gt_y < 0 else gt_y
gt_p = gt_p + 360 if gt_p < 0 else gt_p
gt_r = gt_r + 360 if gt_r < 0 else gt_r

pred_y = pred_y + 360 if pred_y < 0 else pred_y
pred_p = pred_p + 360 if pred_p < 0 else pred_p
pred_r = pred_r + 360 if pred_r < 0 else pred_r

# 输出误差
print(f"GT Yaw: {gt_y}, Pitch: {gt_p}, Roll: {gt_r}")
print(f"Pred Yaw: {pred_y}, Pitch: {pred_p}, Roll: {pred_r}")


