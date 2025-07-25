import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# -----------------------
# 飞行参数
# -----------------------
g = 9.81  # 重力加速度 (m/s^2)
v0 = 26  # 起跳初速度 (m/s)
theta0 = np.radians(11)  # 起跳角度 (°→rad)
T = 2.8  # 模拟时长 (s)
t = np.linspace(0, T, 300)  # 时间点

# 姿态角模拟（alpha: 身体倾角；phi: 滑板开口角）
alpha_0 = np.radians(6.5)
k_alpha = np.radians(0.5)
alpha_t = alpha_0 + k_alpha * t

# -----------------------
# 位移计算
# -----------------------
x = v0 * np.cos(theta0) * t
y = v0 * np.sin(theta0) * t - 0.5 * g * t**2

# 找到落地位置
landing_index = np.where(y >= 0)[0][-1]
landing_x = x[landing_index]
landing_y = y[landing_index]
landing_angle = np.degrees(np.arctan2(
    v0 * np.sin(theta0) - g * t[landing_index],
    v0 * np.cos(theta0)
))

# -----------------------
# 绘图
# -----------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, y, color='royalblue', linewidth=2.5, label="飞行轨迹")

# 姿态箭头（每隔一定时间）
for i in range(0, len(t), 40):
    dx = 2.0 * np.cos(alpha_t[i])
    dy = 2.0 * np.sin(alpha_t[i])
    ax.arrow(x[i], y[i], dx, dy,
             head_width=0.4, head_length=0.6,
             fc='darkorange', ec='darkorange', alpha=0.7)

# 落地标记
ax.plot(landing_x, landing_y, 'ro', label="落地点")
ax.annotate(f"落地角度 {landing_angle:.1f}°",
            xy=(landing_x, landing_y),
            xytext=(landing_x - 12, landing_y + 5),
            arrowprops=dict(facecolor='red', arrowstyle='->'),
            fontsize=12, color='darkred')

# 图像美化
ax.set_title("跳台滑雪：飞行轨迹与姿态变化", fontsize=16)
ax.set_xlabel("水平距离 x (m)", fontsize=13)
ax.set_ylabel("垂直高度 y (m)", fontsize=13)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=12)
ax.set_xlim(0, x[-1] + 5)
ax.set_ylim(0, np.max(y) + 10)
plt.tight_layout()
plt.show()
