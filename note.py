def calculate_A(s1, s2, s3, angle1, angle2, angle3):
    """
    计算表达式 s1*sin(angle1) + s2*sin(angle2 - angle1) + s3*sin(angle3 - angle2 + angle1)

    参数:
    s1, s2, s3: 数值
    angle1, angle2, angle3: 角度（以度为单位）

    返回:
    计算结果
    """
    # 将角度从度转换为弧度
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



def calculate_s(height, weight):
    """
    根据身高和体重估算人体各部位正面投影面积（单位：m²）
    """

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
