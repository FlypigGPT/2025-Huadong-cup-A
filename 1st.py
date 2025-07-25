import math
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# === Step 1: 姿态对应的 Cd 和 A 计算函数 ===
def compute_Cd_A(x1_deg, x2_deg, x3,
                 c0=0.5, c1=0.3, c2=0.2, c3=0.1,
                 a0=0.3, a1=0.1, a2=0.1, a3=0.05):
    # 转换角度为弧度
    x1 = math.radians(x1_deg)  # 身体-大腿角
    x2 = math.radians(x2_deg)  # 大腿-小腿角

    # 空气阻力系数 Cd
    Cd = c0 + c1 * math.sin(x1) + c2 * (1 - math.cos(x2)) + c3 * x3

    # 投影面积 A
    A = a0 + a1 * math.cos(x1) + a2 * math.sin(x2) + a3 * x3

    return Cd, A


# === Step 2: 输入姿态角度 ===
x1_deg = 30  # 身体与大腿夹角
x2_deg = 30  # 大腿与小腿夹角
x3 = 0  # 手臂状态 (可选变量，如是否张开)

Cd, A = compute_Cd_A(x1_deg, x2_deg, x3)
print(f"计算得到的 Cd = {Cd:.3f}")
print(f"计算得到的 A  = {A:.3f} m²")

# === Step 3: 滑行模型参数 ===
g = 9.81  # 重力加速度
theta = math.radians(11)  # 坡度角
mu = 0.03  # 滑雪摩擦系数
rho = 1.225  # 空气密度
m = 70  # 人体质量 kg
s_max = 314.4  # 总滑行长度
ds = 1  # 每一步的滑行距离


# 微分方程 dv/ds
def dvds(s, v):
    Fg = g * math.sin(theta)
    Ff = mu * g * math.cos(theta)
    Fd = 0.5 / m * rho * Cd * A * v ** 2
    return (Fg - Ff - Fd) / v if v != 0 else float('inf')


# 初始条件
s = 0
v = 0.01  # 初始速度（不能为0）

velocities = []
positions = []

# === Step 4: Runge-Kutta 4阶积分计算速度 ===
while s < s_max:
    k1 = dvds(s, v)
    k2 = dvds(s + ds / 2, v + ds * k1 / 2)
    k3 = dvds(s + ds / 2, v + ds * k2 / 2)
    k4 = dvds(s + ds, v + ds * k3)
    v += ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    s += ds
    velocities.append(v)
    positions.append(s)

# 输出结果
print(f"\n末端速度 v_exit ≈ {v:.2f} m/s")

# === Step 5: 可视化 ===
plt.plot(positions, velocities)
plt.xlabel("滑行距离 s (m)")
plt.ylabel("速度 v(s) (m/s)")
plt.title("滑行速度随距离变化图")
plt.grid()
plt.show()
