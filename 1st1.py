#此部分为遗传算法得出的最优角度
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from deap import base, creator, tools, algorithms

matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

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
        "torso_proj_area": round(torso_proj, 4),
        "thigh_proj_area": round(thigh_proj, 4),
        "calf_proj_area": round(calf_proj, 4)
    }

def calculate_A(s1, s2, s3, angle1, angle2, angle3):
    rad_angle1 = math.radians(angle1)
    rad_angle2 = math.radians(angle2)
    rad_angle3 = math.radians(angle3)
    A = (
            s1 * math.sin(rad_angle1) +
            s2 * math.sin(rad_angle2 - rad_angle1) +
            s3 * math.sin(rad_angle3 - rad_angle2 + rad_angle1)
    )
    return A

g = 9.81
theta = math.radians(35)
mu = 0.01
rho = 1.225
m = 70
s_max = 90
ds = 0.01
Cd = 0.7

body_parts = calculate_s(1.8, 80)
s1 = body_parts["torso_proj_area"]
s2 = body_parts["thigh_proj_area"]
s3 = body_parts["calf_proj_area"]

def get_v_exit(angles):
    angle1, angle2, angle3 = angles
    A = calculate_A(s1, s2, s3, angle1, angle2, angle3)
    def dvds(s, v):
        Fg = g * math.sin(theta)
        Ff = mu * g * math.cos(theta)
        Fd = 0.5 / m * rho * Cd * A * v ** 2
        return (Fg - Ff - Fd) / v if v != 0 else float('inf')
    s = 0
    v = 1
    while s < s_max:
        k1 = dvds(s, v)
        k2 = dvds(s + ds / 2, v + ds * k1 / 2)
        k3 = dvds(s + ds / 2, v + ds * k2 / 2)
        k4 = dvds(s + ds, v + ds * k3)
        v += ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        s += ds
    return v

# deap遗传算法部分
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_angle1", np.random.uniform, 0, 90)
toolbox.register("attr_angle2", np.random.uniform, 0, 90)
toolbox.register("attr_angle3", np.random.uniform, 0, 90)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_angle1, toolbox.attr_angle2, toolbox.attr_angle3), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_func(individual):
    angle1, angle2, angle3 = individual
    if not (0 <= angle1 <= 90 and 0 <= angle3 <= 90 and 0 <= angle2 <= 90):
        return -1e6,
    v_exit = get_v_exit([angle1, angle2, angle3])
    return v_exit,

toolbox.register("evaluate", eval_func)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=20)
hof = tools.HallOfFame(1)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=30, halloffame=hof, verbose=False)
best = hof[0]
angle1, angle2, angle3 = best
A = calculate_A(s1, s2, s3, angle1, angle2, angle3)
v_exit = get_v_exit([angle1, angle2, angle3])
print(f"最优角度: angle1={angle1:.2f}°, angle2={angle2:.2f}°, angle3={angle3:.2f}°")
print(f"s1={s1:.4f}, s2={s2:.4f}, s3={s3:.4f}")
print(f"最优A = {A:.4f} m²")
print(f"最优末端速度 v_exit = {v_exit:.2f} m/s")

# 最优角度下的速度曲线可视化
def dvds(s, v):
    Fg = g * math.sin(theta)
    Ff = mu * g * math.cos(theta)
    Fd = 0.5 / m * rho * Cd * A * v ** 2
    return (Fg - Ff - Fd) / v if v != 0 else float('inf')
s = 0
v = 1
velocities = []
positions = []
while s < s_max:
    k1 = dvds(s, v)
    k2 = dvds(s + ds / 2, v + ds * k1 / 2)
    k3 = dvds(s + ds / 2, v + ds * k2 / 2)
    k4 = dvds(s + ds, v + ds * k3)
    v += ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    s += ds
    velocities.append(v)
    positions.append(s)
plt.plot(positions, velocities)
plt.xlabel("滑行距离 s (m)")
plt.ylabel("速度 v(s) (m/s)")
plt.title("滑行速度随距离变化图（最优角度）")
plt.grid()
plt.show()