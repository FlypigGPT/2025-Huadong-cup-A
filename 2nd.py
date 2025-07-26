import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 参数定义
m = 70  # 质量，kg
rho = 1.225  # 空气密度，kg/m^3
g = 9.81  # 重力加速度，m/s^2
A0 = 0.5  # 身体迎风面积基准值，m^2
theta_landing = -35 * np.pi / 180  # 着陆坡倾角

# 滑板迎角和身体俯仰角
alpha_s = 10 * np.pi / 180  # 滑板迎角
alpha_b = 20 * np.pi / 180  # 身体俯仰角

# 动力学函数中的Cd和Cl可以根据角度变化简单设定
def Cd(alpha_s, alpha_b):
    return 0.5 + 0.3 * np.abs(np.sin(alpha_b)) + 0.2 * np.abs(np.sin(alpha_s))

def Cl(alpha_s):
    return 0.6 * np.sin(2 * alpha_s)

def A(alpha_b):
    return A0 * (1 + 0.3 * np.abs(np.sin(alpha_b)))

# 微分方程
def dynamics(t, state):
    vx, vy, x, y = state
    v = np.sqrt(vx**2 + vy**2)
    drag = 0.5 * rho * v * Cd(alpha_s, alpha_b) * A(alpha_b) / m
    lift = 0.5 * rho * v * Cl(alpha_s) * A(alpha_b) / m
    dvx_dt = -drag * vx / v + lift * vy / v
    dvy_dt = -g - drag * vy / v - lift * vx / v
    return [dvx_dt, dvy_dt, vx, vy]

# 初始条件
vx0 = 28 * np.cos(0)  # 初始速度，取平行方向
vy0 = 28 * np.sin(0)
x0, y0 = 0, 3  # 假设起跳点在 y=3 米高处
state0 = [vx0, vy0, x0, y0]

# 定义终止条件（碰到落地斜坡）
def hit_ground(t, y):
    # y = tan(theta_landing) * x 是地面的线性方程
    return y[3] - np.tan(theta_landing) * y[2]

hit_ground.terminal = True
hit_ground.direction = -1

# 求解
sol = solve_ivp(dynamics, [0, 10], state0, t_eval=np.linspace(0, 5, 500), events=hit_ground)

# 提取数据
x = sol.y[2]
y = sol.y[3]

# 作图
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Trajectory')
x_landing = np.linspace(0, 140, 500)
y_landing = np.tan(theta_landing) * x_landing
plt.plot(x_landing, y_landing, '--', label='Landing slope', color='gray')
plt.xlabel('Horizontal distance (m)')
plt.ylabel('Vertical height (m)')
plt.title('Flight Trajectory with Lift and Drag')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
