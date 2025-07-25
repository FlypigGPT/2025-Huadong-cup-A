
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 定义参数
g = 9.81  # 重力加速度 (m/s^2)
m = 70  # 运动员质量 (kg)
mu = 0.02  # 摩擦系数
A = 0.5  # 迎风面积 (m^2)
C_d = 1.0  # 空气阻力系数

# 定义坡度范围
theta = np.linspace(5, 15, 100)  # 坡度从5°到15°
v = np.zeros_like(theta)

# 计算滑行速度
for i, angle in enumerate(theta):
    F_g = m * g * np.sin(np.radians(angle))  # 重力分量
    F_friction = mu * m * g * np.cos(np.radians(angle))  # 摩擦力
    # 使用近似公式来计算速度
    v[i] = np.sqrt((2 * F_g) / (m + (F_friction / (0.5 * 1.225 * A * C_d))))  # 简化模型

# 绘制图形
plt.plot(theta, v, color='b', lw=2)
plt.title("助滑坡坡度与运动员滑行速度的关系", fontsize=14)
plt.xlabel("助滑坡坡度 (°)", fontsize=12)
plt.ylabel("滑行速度 (m/s)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
# 姿势参数：身体前倾角度与起跳速度
alpha1 = np.linspace(0, 20, 100)  # 身体前倾角度
v_optimized = np.sqrt(alpha1 ** 2)  # 简化模型，假设前倾角度与速度的关系

# 绘制图形
plt.plot(alpha1, v_optimized, color='g', lw=2)
plt.title("姿势参数与运动员起跳速度的关系", fontsize=14)
plt.xlabel("身体前倾角度 (°)", fontsize=12)
plt.ylabel("起跳速度 (m/s)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()