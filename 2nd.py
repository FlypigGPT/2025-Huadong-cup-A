import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from deap import base, creator, tools, algorithms
import matplotlib


matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# =====================
# 1. 物理与赛道参数
# =====================
v0 = 28  # 起跳初速度 (m/s)
y0 = 3   # 起跳点高度 (m)
mass = 70  # 质量 (kg)
rho = 1.225  # 空气密度 (kg/m³)
g = 9.81  # 重力加速度 (m/s²)
CD0 = 0.5
CL_max = 0.7
A0 = 1
k1 = 0.8
k2 = 1.2
deltaA = 0.3
jump_angle = 11  # 起跳台仰角 (°)
hill_angle = 34  # 着陆坡角度 (°)
K_point = 120  # K点 (m)
end_point = 140  # 赛道终点 (m)

# =====================
# 2. 地形函数
# =====================
def hill_profile(x):
    return -x * np.tan(np.radians(hill_angle))

# =====================
# 3. 空气动力学系数
# =====================
def compute_CD(alpha_s, alpha_b):
    return CD0 + k1 * alpha_s**2 + k2 * alpha_b**2

def compute_CL(alpha_b):
    alpha_b = np.clip(alpha_b, np.radians(-15), np.radians(15))
    return CL_max * np.sin(2 * alpha_b)

def compute_area(alpha_s):
    return A0 + deltaA * np.sin(alpha_s)

# =====================
# 4. 飞行动力学微分方程
# =====================
def flight_dynamics(t, Y, alpha_s, alpha_b):
    x, y, vx, vy = Y
    v = np.sqrt(vx**2 + vy**2)
    if v == 0:
        return [vx, vy, 0, -g]
    CD = compute_CD(alpha_s, alpha_b)
    CL = compute_CL(alpha_b)
    A = compute_area(alpha_s)
    F_drag = 0.5 * rho * CD * A * v**2
    F_lift = 0.5 * rho * CL * A * v**2
    ax = (-F_drag * vx + F_lift * (-vy)) / (v * mass)
    ay = (-F_drag * vy + F_lift * vx) / (v * mass) - g
    return [vx, vy, ax, ay]

# =====================
# 5. 落地事件检测
# =====================
def hit_ground(t, Y, *args):
    x, y, vx, vy = Y
    return y - hill_profile(x)
hit_ground.terminal = True
hit_ground.direction = -1

# =====================
# 6. 单次跳跃仿真
# =====================
def simulate_jump(alpha_s, alpha_b):
    theta0 = np.radians(jump_angle) + alpha_s
    vx0 = v0 * np.cos(theta0)
    vy0 = v0 * np.sin(theta0)
    Y0 = [0, y0, vx0, vy0]
    sol = solve_ivp(flight_dynamics, (0, 10), Y0, args=(alpha_s, alpha_b), events=hit_ground, max_step=0.01)
    if sol.status != 1:
        return None
    x, y = sol.y[0][-1], sol.y[1][-1]
    vx, vy = sol.y[2][-1], sol.y[3][-1]
    landing_angle = np.degrees(np.arctan2(vy, vx))  # 与水平夹角，负值向下
    speed = np.sqrt(vx**2 + vy**2)
    return {
        "x": x, "y": y,
        "vx": vx, "vy": vy,
        "angle": landing_angle,
        "speed": speed,
        "sol": sol
    }

# =====================
# 7. 评分函数
# =====================
def score_function(x, landing_angle):
    # 距离分
    if x > end_point:
        distance_score = 0
    else:
        distance_score = 120 + (x - K_point) * 1.8
        distance_score = max(distance_score, 0)
    # 姿势分（落地角度与坡度越接近越好，满分10分，每离34° 5度扣1分）
    angle_diff = abs(landing_angle + hill_angle)
    posture_score = 10 - angle_diff / 5  # 每5度扣1分，满分10分，可为负
    return distance_score, posture_score

# =====================
# 8. 遗传算法优化
# =====================
print("开始遗传算法优化")

def fitness_func(individual):
    alpha_s, alpha_b = individual
    sim = simulate_jump(alpha_s, alpha_b)
    if sim is None:
        return -1e6,
    x = sim["x"]
    angle = sim["angle"]
    distance_score, posture_score = score_function(x, angle)
    total_score = distance_score + posture_score
    return total_score,

# 创建遗传算法工具箱
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# 注册基因生成函数
toolbox.register("attr_alpha_s", np.random.uniform, 0, np.radians(30))
toolbox.register("attr_alpha_b", np.random.uniform, np.radians(-15), np.radians(15))
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_alpha_s, toolbox.attr_alpha_b), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册遗传操作
toolbox.register("evaluate", fitness_func)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=np.radians(2), indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行遗传算法
pop = toolbox.population(n=30)  # 减少种群大小
hof = tools.HallOfFame(1)
algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=30, halloffame=hof, verbose=True)  # 减少代数

# 获取最优解
best = hof[0]
alpha_s_opt, alpha_b_opt = best
best_result = simulate_jump(alpha_s_opt, alpha_b_opt)

print(f"✅ 遗传算法优化完成！\n")

# =====================
# 9. 输出最优解
# =====================
print("🏆 最优解：")
print(f"俯仰角 α_s: {np.degrees(alpha_s_opt):.2f}°")
print(f"攻角 α_b: {np.degrees(alpha_b_opt):.2f}°")
print(f"落地点 x: {best_result['x']:.2f} m")
print(f"落地角度: {best_result['angle']:.2f}°")
print(f"落地速度: {best_result['speed']:.2f} m/s")

# 计算分数
distance_score, posture_score = score_function(best_result['x'], best_result['angle'])
total_score = distance_score + posture_score
print(f"距离分: {distance_score:.2f}")
print(f"姿势分: {posture_score:.2f}")
print(f"总分: {total_score:.2f}")

# =====================
# 10. 可视化最优轨迹
# =====================
sol = best_result["sol"]
x_vals, y_vals = sol.y[0], sol.y[1]
x_hill = np.linspace(0, max(x_vals)+10, 300)
y_hill = hill_profile(x_hill)
plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, label="飞行轨迹")
plt.plot(x_hill, y_hill, 'k--', label="着陆坡 (34°)")
plt.plot(best_result["x"], best_result["y"], 'ro', label="着陆点")
plt.title("跳台滑雪轨迹（遗传算法）")
plt.xlabel("水平距离(m)")
plt.ylabel("高度 (m)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()
