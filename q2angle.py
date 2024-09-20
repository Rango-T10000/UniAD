import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

gt = [-0.7504501527141022, -0.0076295847961364415, 0.00847103369020136, -0.6608287216181199]
pred = [-0.743742073537717, -0.007726799292098994, 0.008375154888041067, -0.6683695694771852]

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

# 计算绝对误差
abs_error_y = abs(gt_y - pred_y)
abs_error_p = abs(gt_p - pred_p)
abs_error_r = abs(gt_r - pred_r)

# 计算均方误差
mse_y = np.square(gt_y - pred_y)
mse_p = np.square(gt_p - pred_p)
mse_r = np.square(gt_r - pred_r)

# 输出误差
print(f"GT Yaw: {gt_y}, Pitch: {gt_p}, Roll: {gt_r}")
print(f"Pred Yaw: {pred_y}, Pitch: {pred_p}, Roll: {pred_r}")
print(f"Absolute Error (Yaw, Pitch, Roll): {abs_error_y}, {abs_error_p}, {abs_error_r}")
print(f"Mean Squared Error (Yaw, Pitch, Roll): {mse_y}, {mse_p}, {mse_r}")

# 平均误差
avg_abs_error = np.mean([abs_error_y, abs_error_p, abs_error_r])
avg_mse = np.mean([mse_y, mse_p, mse_r])

print(f"Average Absolute Error: {avg_abs_error}")
print(f"Average Mean Squared Error: {avg_mse}")