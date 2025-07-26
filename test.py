import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === 参数设置 ===
v0 = 28  # 起跳速度 m/s
theta0 = np.radians(10)  # 起跳角度（与水平线夹角）
y0 = 3  # 起跳点高度
mass = 70  # 质量 kg
rho = 1.225  # 空气密度 kg/m³
g = 9.81  # 重力加速度
Cd = 0.5  # 阻力系数（可调整）
A = 0.5  # 迎风面积（可调整）

# === 地形函数：34°下降直线 ===
def hill_profile(x):
    return y0 - x * np.tan(np.radians(34))

# === 微分方程：二维有阻力运动 ===
def flight_dynamics(t, Y):
    x, y, vx, vy = Y
    v = np.sqrt(vx**2 + vy**2)
    F_drag = 0.5 * rho * Cd * A * v**2
    ax = -F_drag * vx / (mass * v)
    ay = -g - F_drag * vy / (mass * v)
    return [vx, vy, ax, ay]

# === 事件函数：飞行轨迹与地形相交 ===
def hit_ground(t, Y):
    x, y, vx, vy = Y
    return y - hill_profile(x)
hit_ground.terminal = True
hit_ground.direction = -1  # 从正到负穿过0，表示从空中到地面

# === 初始条件 ===
vx0 = v0 * np.cos(theta0)
vy0 = v0 * np.sin(theta0)
Y0 = [0, y0, vx0, vy0]

# === 求解微分方程 ===
t_span = (0, 10)
sol = solve_ivp(flight_dynamics, t_span, Y0, events=hit_ground, max_step=0.01, rtol=1e-8)

# === 提取数据 ===
x_vals, y_vals = sol.y[0], sol.y[1]

# === 计算落地时倾角 ===
vx_f, vy_f = sol.y[2][-1], sol.y[3][-1]
landing_angle_deg = np.degrees(np.arctan2(vy_f, vx_f))

# === 重新绘制地形线 ===
x_hill = np.linspace(0, max(x_vals) + 5, 300)
y_hill = hill_profile(x_hill)

# === 可视化 ===
plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, label="Flight Path", linewidth=2)
plt.plot(x_hill, y_hill, 'k--', label="Hill (34°)", linewidth=2)
plt.plot(x_vals[-1], y_vals[-1], 'ro', label="Landing Point")
plt.title("Ski Jump Trajectory with Air Drag", fontsize=14)
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Vertical Height (m)")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

# === 输出落地信息 ===
landing_info = {
    "落地点 x (m)": round(x_vals[-1], 2),
    "落地点 y (m)": round(y_vals[-1], 2),
    "落地速度 (m/s)": round(np.sqrt(vx_f**2 + vy_f**2), 2),
    "落地倾角 (°)": round(landing_angle_deg, 2)
}
print(landing_info)
