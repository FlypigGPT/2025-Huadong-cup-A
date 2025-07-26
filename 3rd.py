import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# =====================
# 1. 着陆平衡模型参数
# =====================
# 物理参数
mass = 70  # 运动员质量 (kg)
g = 9.81  # 重力加速度 (m/s²)
hill_angle = 34  # 着陆坡角度 (°)

# 假设的落地条件（从飞行阶段得到）
landing_speed = 34  # 落地速度 (m/s)
flight_distance = 120  # 飞行距离 (m)

# =====================
# 2. 着陆平衡评分函数
# =====================
def landing_balance_score(landing_angle, body_posture, arm_position):
    """
    着陆平衡评分函数（总分20分）
    landing_angle: 落地角度（与水平面夹角，负值向下）
    body_posture: 身体姿态 (0=完全蹲伏, 1=直立)
    arm_position: 手臂位置 (0=下垂, 1=完全张开)
    """
    # 1. 角度匹配评分（10分）
    angle_diff = abs(landing_angle + hill_angle)
    angle_score = max(0, 10 - angle_diff)  # 每偏差1°扣1分，最多10分
    
    # 2. 身体姿态评分（6分）
    posture_score = (1 - body_posture) * 6  # 0=蹲伏得6分，1=直立得0分
    
    # 3. 手臂平衡评分（2分）
    arm_score = arm_position * 2  # 0=下垂得0分，1=完全张开得2分
    
    # 4. 速度惩罚（最多2分）
    speed_penalty = min(abs(landing_speed - 25) * 0.2, 2)  # 每偏离1m/s扣0.2分，最多扣2分
    
    # 5. 综合平衡评分
    total_score = angle_score + posture_score + arm_score - speed_penalty
    total_score = max(0, min(20, total_score))  # 保证在0~20分
    
    return {
        "total_score": total_score,
        "angle_score": angle_score,
        "posture_score": posture_score,
        "arm_score": arm_score,
        "speed_penalty": speed_penalty,
        "angle_diff": angle_diff
    }

# =====================
# 3. 着陆安全性分析
# =====================
def landing_safety_analysis(landing_angle, body_posture, arm_position):
    """
    着陆安全性分析
    """
    # 冲击力计算（简化模型）
    impact_force = mass * landing_speed * np.cos(np.radians(abs(landing_angle + hill_angle)))
    
    # 缓冲效果（蹲伏减少冲击）
    buffer_factor = 1 - 0.6 * body_posture  # 蹲伏减少60%冲击
    effective_impact = impact_force * buffer_factor
    
    # 平衡稳定性（角度匹配度影响）
    stability_factor = max(0, 1 - abs(landing_angle + hill_angle) / 30)
    
    # 摔倒风险评估
    fall_risk = 1 - stability_factor * (1 - body_posture) * (0.5 + 0.5 * arm_position)
    
    return {
        "impact_force": impact_force,
        "effective_impact": effective_impact,
        "stability_factor": stability_factor,
        "fall_risk": fall_risk
    }

# =====================
# 4. 二维参数扫描优化
# =====================
print("开始着陆平衡策略优化...")

# 定义参数范围
landing_angle_vals = np.linspace(-50, -20, 40)  # 落地角度范围
body_posture_vals = np.linspace(0, 1, 40)      # 身体姿态范围

# 创建网格
ANGLE, POSTURE = np.meshgrid(landing_angle_vals, body_posture_vals)
SCORE = np.zeros_like(ANGLE)
SAFETY = np.zeros_like(ANGLE)

# 遍历参数组合
for i in range(ANGLE.shape[0]):
    for j in range(ANGLE.shape[1]):
        # 固定手臂位置为最优值（张开）
        arm_pos = 1.0
        
        # 计算平衡评分
        score_result = landing_balance_score(ANGLE[i, j], POSTURE[i, j], arm_pos)
        SCORE[i, j] = score_result["total_score"]
        
        # 计算安全性指标
        safety_result = landing_safety_analysis(ANGLE[i, j], POSTURE[i, j], arm_pos)
        SAFETY[i, j] = safety_result["stability_factor"]

# =====================
# 5. 三维可视化
# =====================
fig = plt.figure(figsize=(16, 6))

# 子图1：平衡评分
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(ANGLE, POSTURE, SCORE, cmap=cm.viridis, 
                        edgecolor='k', linewidth=0.2, alpha=0.8)

# 找到最优点
max_idx = np.unravel_index(np.argmax(SCORE), SCORE.shape)
opt_angle = ANGLE[max_idx]
opt_posture = POSTURE[max_idx]
opt_score = SCORE[max_idx]

# 标记最优点
ax1.scatter(opt_angle, opt_posture, opt_score, color='red', s=200, 
           marker='*', label=f'最优点\n角度: {opt_angle:.1f}°\n姿态: {opt_posture:.2f}\n分数: {opt_score:.1f}')

ax1.set_title("着陆平衡评分", fontsize=14, pad=20)
ax1.set_xlabel("落地角度 (°)", fontsize=12)
ax1.set_ylabel("身体姿态 (0=蹲伏, 1=直立)", fontsize=12)
ax1.set_zlabel("平衡评分", fontsize=12)
ax1.view_init(elev=25, azim=45)
ax1.legend()

# 子图2：稳定性分析
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(ANGLE, POSTURE, SAFETY, cmap=cm.plasma, 
                        edgecolor='k', linewidth=0.2, alpha=0.8)

# 标记最稳定点
max_stable_idx = np.unravel_index(np.argmax(SAFETY), SAFETY.shape)
stable_angle = ANGLE[max_stable_idx]
stable_posture = POSTURE[max_stable_idx]
stable_safety = SAFETY[max_stable_idx]

ax2.scatter(stable_angle, stable_posture, stable_safety, color='green', s=200, 
           marker='o', label=f'最稳定点\n角度: {stable_angle:.1f}°\n姿态: {stable_posture:.2f}\n稳定性: {stable_safety:.2f}')

ax2.set_title("着陆稳定性分析", fontsize=14, pad=20)
ax2.set_xlabel("落地角度 (°)", fontsize=12)
ax2.set_ylabel("身体姿态 (0=蹲伏, 1=直立)", fontsize=12)
ax2.set_zlabel("稳定性因子", fontsize=12)
ax2.view_init(elev=25, azim=45)
ax2.legend()

plt.tight_layout()
plt.show()

# =====================
# 6. 输出最优量化动作建议和安全性分析
# =====================
print(f"\n\n落地速度：{landing_speed:.2f} m/s （此为飞行阶段影响）")
print(f"着陆阶段最优动作参数：")
print(f"最优落地角度：{opt_angle:.2f}° （与坡度夹角：{abs(opt_angle + hill_angle):.2f}°）")
print(f"最优蹲伏程度：{opt_posture:.2f} （0=完全蹲伏，1=直立）")
print(f"最优手臂张开度：1.00 ")

# 进一步量化膝关节弯曲角度（假设0=120°，1=180°）
knee_angle = 120 + 60 * opt_posture  # 0=120°, 1=180°
print(f"5. 膝关节弯曲角度：{knee_angle:.0f}° ")

# 详细分析
print(f"\n详细评分：")
detailed_score = landing_balance_score(opt_angle, opt_posture, 1.0)
print(f"角度匹配分: {detailed_score['angle_score']:.1f}/10")
print(f"姿态缓冲分: {detailed_score['posture_score']:.1f}/6")
print(f"手臂平衡分: {detailed_score['arm_score']:.1f}/2")
print(f"速度惩罚: -{detailed_score['speed_penalty']:.1f}")

print(f"\n安全性分析：")
detailed_safety = landing_safety_analysis(opt_angle, opt_posture, 1.0)
print(f"冲击力: {detailed_safety['impact_force']:.0f} N")
print(f"有效冲击: {detailed_safety['effective_impact']:.0f} N")
print(f"稳定性因子: {detailed_safety['stability_factor']:.2f}")
print(f"摔倒风险: {detailed_safety['fall_risk']:.1%}")
