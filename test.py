import math
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


#输入身高和体重来计算s1,s2,s3
def calculate_s(height, weight):
    # 全身表面积估算（Du Bois公式）
    BSA = 0.007184 * (height * 100) ** 0.725 * (weight) ** 0.425  # m²

    # 比例估算每个部位所占体表面积（文献参考估计值）
    # 头+躯干约占 55%
    # 大腿约占 21%
    # 小腿约占 14%

    torso_area = BSA * 0.55
    thigh_area = BSA * 0.21
    calf_area = BSA * 0.14

    # 正面投影约为体表面积的一半（粗略估算）
    torso_proj = torso_area * 0.5
    thigh_proj = thigh_area * 0.5
    calf_proj = calf_area * 0.5

    return {
        "torso_proj_area": round(torso_proj, 4),  # m²
        "thigh_proj_area": round(thigh_proj, 4),  # m²
        "calf_proj_area": round(calf_proj, 4)     # m²
    }



# === Step 1: 姿态对应的 A 计算函数 ===
def calculate_A(s1, s2, s3, angle1, angle2, angle3):
    rad_angle1 = math.radians(angle1)
    rad_angle2 = math.radians(angle2)
    rad_angle3 = math.radians(angle3)

    # 计算表达式
    A = (
            s1 * math.sin(rad_angle1) +
            s2 * math.sin(rad_angle2 - rad_angle1) +
            s3 * math.sin(rad_angle3 - rad_angle2 + rad_angle1)
    )

    return A


# === 滑行模型参数 ===
g = 9.81  # 重力加速度
theta = math.radians(35)  # 坡度角
mu = 0.01  # 滑雪摩擦系数
rho = 1.225  # 空气密度
m = 70  # 人体质量 kg
s_max = 90  # 总滑行长度
ds = 0.01  # 每一步的滑行距离
Cd = 0.7


# === Step 2: 输入姿态角度 ===
body_parts = calculate_s(1.8, 80)
s1 = body_parts["torso_proj_area"]
s2 = body_parts["thigh_proj_area"]
s3 = body_parts["calf_proj_area"]


angle1 = 30  # 身体与大腿夹角
angle2 = 30  # 大腿与小腿夹角
angle3 = 30  # 手臂状态 (可选变量，如是否张开)

A = calculate_A(s1,s2,s3,angle1,angle2,angle3)
print(s1)
print(s2)
print(s3)
print(f"假设的Cd = {Cd:.3f}")
print(f"计算得到的 A  = {A:.3f} m²")


# 微分方程 dv/ds
def dvds(s, v):
    Fg = g * math.sin(theta)
    Ff = mu * g * math.cos(theta)
    Fd = 0.5 / m * rho * Cd * A * v ** 2
    return (Fg - Ff - Fd) / v if v != 0 else float('inf')


# 初始条件
s = 0
v = 1  # 初始速度（不能为0）

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