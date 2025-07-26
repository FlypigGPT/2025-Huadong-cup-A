import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# -----------------------
# 飞行轨迹模拟（含动态姿态）
# -----------------------
g = 9.81  # 重力加速度
v0 = 33.4  # 起跳初速度（单位 m/s）
theta0 = np.radians(11)  # 起跳仰角（单位 ° → rad）

T = 2.8  # 飞行总时长估计
t = np.linspace(0, T, 300)

# 飞行姿态角变化（α：身体姿态角；φ：雪板角）
alpha_0 = np.radians(6.5)
k_alpha = np.radians(0.5)
phi_0 = np.radians(8)
k_phi = np.radians(0.6)

alpha_t = alpha_0 + k_alpha * t
phi_t = phi_0 + k_phi * t

# 飞行轨迹
x = v0 * np.cos(theta0) * t
y = v0 * np.sin(theta0) * t - 0.5 * g * t**2

# 找落地点
landing_index = np.where(y >= 0)[0][-1]
landing_x = x[landing_index]
landing_y = y[landing_index]
landing_angle = np.degrees(np.arctan2(
    v0 * np.sin(theta0) - g * t[landing_index],
    v0 * np.cos(theta0)
))

# -----------------------
# 可视化轨迹（姿态 + 着陆）
# -----------------------
fig = plt.figure(figsize=(12, 7))
ax = plt.subplot(1, 1, 1)

# 带速度着色的轨迹线
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
speed = np.sqrt((v0 * np.cos(theta0))**2 + (v0 * np.sin(theta0) - g * t)**2)
norm = plt.Normalize(speed.min(), speed.max())
lc = LineCollection(segments, cmap='plasma', norm=norm)
lc.set_array(speed)
lc.set_linewidth(3)
ax.add_collection(lc)

# 姿态箭头（简化模拟姿态变化）
for i in range(0, len(t), 50):
    ax.arrow(x[i], y[i],
             2 * np.cos(alpha_t[i]), 2 * np.sin(alpha_t[i]),
             head_width=0.4, head_length=0.6,
             fc='teal', ec='teal', alpha=0.7)

# 落地点标注
ax.plot(landing_x, landing_y, 'ro')
ax.annotate(f"落地角 ≈ {landing_angle:.1f}°",
            xy=(landing_x, landing_y),
            xytext=(landing_x - 10, landing_y + 5),
            arrowprops=dict(facecolor='red', arrowstyle='->'),
            fontsize=12, color='darkred')

# 色条：速度
cb = fig.colorbar(lc, ax=ax)
cb.set_label('飞行速度 v (m/s)', fontsize=12)

# 坐标轴 & 标题
ax.set_title("跳台滑雪飞行轨迹（含姿态箭头）", fontsize=16)
ax.set_xlabel("水平距离 x (m)", fontsize=12)
ax.set_ylabel("垂直高度 y (m)", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(0, x[-1] + 5)
ax.set_ylim(0, np.max(y) + 10)
plt.tight_layout()
plt.show()
