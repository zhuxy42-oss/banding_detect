import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# 创建数据
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

# 计算到原点的距离
R = np.sqrt(X**2 + Y**2)

# 创建倒立圆锥：z = 1 - r
Z = 1 - R

# 将 z < 0.1 的部分设为 0.1
Z[Z < 0.1] = 0.1

# 定义四个点的xy坐标
points_xy = [(0, 0), (0.5, 0), (0, 0.5), (0.25, 0.25)]

# 计算这些点在曲面上的z坐标
points_3d = []
for x, y in points_xy:
    r = np.sqrt(x**2 + y**2)
    z = 1 - r
    if z < 0.1:
        z = 0.1
    points_3d.append((x, y, z))
points_3d = np.array(points_3d)

def cone_geodesic(start_point, end_point, num_points=50):
    """
    计算圆锥曲面上的测地线
    """
    # 将笛卡尔坐标转换为极坐标
    def cartesian_to_polar(x, y, z):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta, z
    
    # 将极坐标转换为笛卡尔坐标
    def polar_to_cartesian(r, theta, z):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, z
    
    start_r, start_theta, start_z = cartesian_to_polar(*start_point)
    end_r, end_theta, end_z = cartesian_to_polar(*end_point)
    
    # 处理角度环绕
    theta_diff = end_theta - start_theta
    if theta_diff > np.pi:
        theta_diff -= 2 * np.pi
    elif theta_diff < -np.pi:
        theta_diff += 2 * np.pi
    
    # 在极坐标中线性插值
    r_points = np.linspace(start_r, end_r, num_points)
    theta_points = np.linspace(start_theta, start_theta + theta_diff, num_points)
    
    # 转换回笛卡尔坐标
    geodesic_points = []
    for r, theta in zip(r_points, theta_points):
        z = 1 - r  # 圆锥曲面方程
        if z < 0.1:
            z = 0.1
        x, y, z = polar_to_cartesian(r, theta, z)
        geodesic_points.append([x, y, z])
    
    return np.array(geodesic_points)

def improved_cone_geodesic(start_point, end_point, num_points=100):
    """
    改进的测地线计算方法，考虑圆锥的展开
    """
    # 将点投影到圆锥展开面
    def cone_to_developed(x, y, z):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        # 圆锥展开后的半径（弧长）
        L = r * np.sqrt(2)  # 对于45度圆锥，斜高系数为√2
        return L, theta
    
    def developed_to_cone(L, theta):
        r = L / np.sqrt(2)  # 反变换
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 1 - r
        if z < 0.1:
            z = 0.1
        return x, y, z
    
    start_L, start_theta = cone_to_developed(*start_point)
    end_L, end_theta = cone_to_developed(*end_point)
    
    # 在展开面上计算直线（测地线）
    L_points = np.linspace(start_L, end_L, num_points)
    theta_points = np.linspace(start_theta, end_theta, num_points)
    
    # 转换回圆锥坐标
    geodesic_points = []
    for L, theta in zip(L_points, theta_points):
        x, y, z = developed_to_cone(L, theta)
        geodesic_points.append([x, y, z])
    
    return np.array(geodesic_points)

# 创建图形
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, 
                      linewidth=0, antialiased=True)

# 绘制四个点
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
           color='red', s=100, label='投影点', zorder=5)

# 在点(0.25,0.25)处添加特殊标记
ax.scatter([points_3d[3, 0]], [points_3d[3, 1]], [points_3d[3, 2]], 
           color='yellow', s=200, marker='*', label='中心点(0.25,0.25)', zorder=6)

# 计算并绘制测地线
# 连接中心点(0.25,0.25)到侧面点1(0.5,0)
geodesic1 = improved_cone_geodesic(points_3d[3], points_3d[1], 100)
ax.plot(geodesic1[:, 0], geodesic1[:, 1], geodesic1[:, 2], 
        color='red', linewidth=4, label='测地线1: (0.25,0.25)→(0.5,0)', zorder=4)

# 连接中心点(0.25,0.25)到侧面点2(0,0.5)
geodesic2 = improved_cone_geodesic(points_3d[3], points_3d[2], 100)
ax.plot(geodesic2[:, 0], geodesic2[:, 1], geodesic2[:, 2], 
        color='blue', linewidth=4, label='测地线2: (0.25,0.25)→(0,0.5)', zorder=4)

# 可选：连接中心点到顶点
geodesic3 = improved_cone_geodesic(points_3d[3], points_3d[0], 100)
ax.plot(geodesic3[:, 0], geodesic3[:, 1], geodesic3[:, 2], 
        color='green', linewidth=3, linestyle='--', alpha=0.7, 
        label='测地线3: (0.25,0.25)→(0,0)', zorder=4)

# 绘制原始三角形（虚线）
triangle = np.array([points_3d[0], points_3d[1], points_3d[2], points_3d[0]])
ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
        color='black', linewidth=2, linestyle=':', alpha=0.5, 
        label='原始三角形', zorder=3)

# 设置坐标轴标签
ax.set_xlabel('X轴', fontsize=12)
ax.set_ylabel('Y轴', fontsize=12)
ax.set_zlabel('Z轴', fontsize=12)

# 设置标题
ax.set_title('圆锥曲面上的测地线连接', fontsize=16, pad=20)

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Z值')

# 设置视角
ax.view_init(elev=25, azim=45)

# 设置坐标轴范围
ax.set_xlim([-0.6, 0.6])
ax.set_ylim([-0.6, 0.6])
ax.set_zlim([0, 1.1])

# 添加图例和网格
ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印测地线信息
print("测地线连接信息：")
print(f"测地线1: ({points_3d[3,0]:.2f}, {points_3d[3,1]:.2f}, {points_3d[3,2]:.2f}) → ({points_3d[1,0]:.2f}, {points_3d[1,1]:.2f}, {points_3d[1,2]:.2f})")
print(f"测地线2: ({points_3d[3,0]:.2f}, {points_3d[3,1]:.2f}, {points_3d[3,2]:.2f}) → ({points_3d[2,0]:.2f}, {points_3d[2,1]:.2f}, {points_3d[2,2]:.2f})")
print(f"测地线长度1: {np.sum(np.linalg.norm(np.diff(geodesic1, axis=0), axis=1)):.3f}")
print(f"测地线长度2: {np.sum(np.linalg.norm(np.diff(geodesic2, axis=0), axis=1)):.3f}")

# 显示测地线的细节
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': '3d'})

# 左侧：测地线1的细节
surf2 = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)
ax2.scatter(points_3d[[1,3], 0], points_3d[[1,3], 1], points_3d[[1,3], 2], 
           color=['red', 'yellow'], s=100)
ax2.scatter([points_3d[3, 0]], [points_3d[3, 1]], [points_3d[3, 2]], 
           color='yellow', s=200, marker='*')
ax2.plot(geodesic1[:, 0], geodesic1[:, 1], geodesic1[:, 2], 
        color='red', linewidth=5, label='测地线1')
ax2.set_title('测地线1: (0.25,0.25)→(0.5,0)', fontsize=14)
ax2.view_init(elev=20, azim=30)

# 右侧：测地线2的细节
surf3 = ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)
ax3.scatter(points_3d[[2,3], 0], points_3d[[2,3], 1], points_3d[[2,3], 2], 
           color=['blue', 'yellow'], s=100)
ax3.scatter([points_3d[3, 0]], [points_3d[3, 1]], [points_3d[3, 2]], 
           color='yellow', s=200, marker='*')
ax3.plot(geodesic2[:, 0], geodesic2[:, 1], geodesic2[:, 2], 
        color='blue', linewidth=5, label='测地线2')
ax3.set_title('测地线2: (0.25,0.25)→(0,0.5)', fontsize=14)
ax3.view_init(elev=20, azim=60)

plt.tight_layout()
plt.savefig('sf.png', dpi=300)
plt.show()