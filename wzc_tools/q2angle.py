import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

#下面两个数值一样，符号相反的四元数所表示的角度一模一样！
#注意用四元数举例子不用胡乱举例，
gt = [-0.7504501527141022, -0.0076295847961364415, 0.00847103369020136, -0.6608287216181199]
pred = [0.7504501527141022, 0.0076295847961364415, -0.00847103369020136, 0.6608287216181199]

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


#-----------------------------------------直接计算两个四元数的点积----------------------------------
def quaternion_dot_and_angle(gt, pred):
    # 确保输入是numpy数组
    gt = np.array(gt)
    pred = np.array(pred)

    # 计算点积
    dot_product = np.dot(gt, pred)

    # 限制点积在 [-1, 1] 的范围内
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 计算角度差（以弧度为单位）
    angle = 2 * np.arccos(abs(dot_product))

    return dot_product, angle

# 计算
dot_product, angle = quaternion_dot_and_angle(gt, pred)
print("Dot Product:", dot_product)
print("Angle (radians):", angle)
print("Angle (degrees):", np.degrees(angle))