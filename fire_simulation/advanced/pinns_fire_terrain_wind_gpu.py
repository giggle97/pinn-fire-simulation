import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义物理参数与场景配置
# ==========================================
# 模拟区域: 10m x 10m
X_MIN, Y_MIN = 0, 0
X_MAX, Y_MAX = 10, 10

# 基础燃烧速度 (m/s)
R_BASE = 1.0

# 环境影响系数
K_SLOPE = 3.0   # 坡度影响系数 (值越大，坡度对速度影响越显著)
K_WIND = 2.0    # 风影响系数

# 风场设置 (单位向量 + 风速)
# 假设风从左下吹向右上 (45度角)
WIND_DIR = np.array([1.0, 1.0])
WIND_DIR = WIND_DIR / np.linalg.norm(WIND_DIR) # 归一化
WIND_SPEED_FACTOR = 1.5 # 风的强度因子

# ==========================================
# 2. 定义地形函数 (小山坡)
# ==========================================
def get_slope_field(x_np):
    """
    计算给定坐标点的坡度值。
    地形假设：中心 (5,5) 有一个高斯形状的小山丘。
    返回：坡度标量场 (0.0 ~ 1.0)，代表坡度的陡峭程度。
    """
    # x_np shape: [N, 2]
    cx, cy = 5.0, 5.0
    # 计算距离中心的距离
    dist = np.sqrt((x_np[:, 0:1] - cx)**2 + (x_np[:, 1:2] - cy)**2)
    
    # 高斯函数模拟山丘高度: h = exp(-d^2 / sigma^2)
    sigma = 2.0
    height = np.exp(-(dist**2) / (2 * sigma**2))
    
    # 坡度近似为高度的梯度幅值 (这里用解析解简化模拟)
    # 对于高斯函数，坡度在山顶为0，在山腰最大，在山脚趋于0
    slope_mag = (dist / sigma**2) * height
    
    # 归一化到 0-1 之间方便调整系数
    max_slope = 0.5 / sigma # 理论最大值近似
    slope_normalized = np.clip(slope_mag / (max_slope + 1e-6), 0, 1)
    
    return slope_normalized

# ==========================================
# 3. 定义偏微分方程 (PDE) - 核心物理逻辑
# ==========================================
def pde(x, y):
    """
    x: 输入坐标 [N, 2] (x, y)
    y: 网络预测的到达时间 [N, 1] (t)
    
    方程: |grad(t)| = 1 / R(x, y, slope, wind)
    """
    # 1. 计算时间的梯度 (自动微分)
    dy_dx = dde.grad.jacobian(y, x, i=0, j=0)
    dy_dy = dde.grad.jacobian(y, x, i=0, j=1)
    
    # 梯度模长 |grad(t)|
    grad_mag = torch.sqrt(dy_dx**2 + dy_dy**2 + 1e-8)
    
    # 2. 计算传播方向单位向量 n = grad(t) / |grad(t)|
    # 注意：grad(t) 指向时间增加最快的方向，即火的传播方向
    nx = dy_dx / grad_mag
    ny = dy_dy / grad_mag
    
    # 3. 获取当前位置的环境参数
    # DeepXDE 的 x 是 Tensor，需要转为 numpy 计算地形，再转回 tensor
    # 在实际训练中，dde.grad 会处理图，但这里为了调用外部 numpy 函数 get_slope_field
    # 我们需要确保操作是可导的或者使用 dde 内置操作。
    # 为了简单且保持可导性，我们直接在 torch 中重写坡度计算逻辑：
    
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    cx, cy = 5.0, 5.0
    dist_sq = (x_coord - cx)**2 + (y_coord - cy)**2
    sigma = 2.0
    # 高度 h
    h = torch.exp(-dist_sq / (2 * sigma**2))
    # 坡度 magnitude (解析导数)
    # d(h)/dr = -r/sigma^2 * h -> 坡度大小取绝对值
    slope_mag = torch.sqrt(dist_sq) / (sigma**2) * h
    # 简单归一化因子 (近似)
    slope_val = torch.clamp(slope_mag * 2.5, 0, 1) # 2.5 是经验缩放因子
    
    # 4. 计算风向投影 (顺风/逆风)
    # cos_theta = n · wind_dir
    cos_theta = nx * WIND_DIR[0] + ny * WIND_DIR[1]
    
    # 5. 构建动态速度场 R(x,y)
    # 逻辑：R = R_base * (1 + K_slope * slope) * (1 + K_wind * cos_theta)
    # 解释：
    # - 坡度越大 (slope_val 大)，速度越快 (模拟上坡助燃)
    # - cos_theta > 0 (顺风) 速度增加; < 0 (逆风) 速度减小
    R = R_BASE * (1.0 + K_SLOPE * slope_val) * (1.0 + K_WIND * cos_theta)
    
    # 限制最小速度，防止除以零或负数
    R = torch.clamp(R, min=0.2)
    
    # 6. 返回 PDE 残差: |grad(t)| - 1/R = 0
    residual = grad_mag - 1.0 / R
    
    return residual

# ==========================================
# 4. 定义几何空间与边界条件
# ==========================================
geom = dde.geometry.Rectangle([X_MIN, Y_MIN], [X_MAX, Y_MAX])

# 起火点：左下角 (1, 1)
def boundary_fire_source(x, on_boundary):
    dist = np.sqrt((x[0] - 1.0)**2 + (x[1] - 1.0)**2)
    return dist < 0.4  # 半径稍小一点，让波形更清晰

ic = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_fire_source)

# ==========================================
# 5. 构建数据集
# ==========================================
# 增加采样点以捕捉复杂的坡度变化
data = dde.data.PDE(
    geom, 
    pde, 
    ic, 
    num_domain=5000,      # 内部点增多
    num_boundary=800,     # 边界点增多
    train_distribution="pseudo",
    num_test=2000
)

# ==========================================
# 6. 定义神经网络
# ==========================================
# 结构：2 -> 60 -> 60 -> 60 -> 60 -> 1
# 加深网络以拟合非线性的速度和地形交互
net = dde.nn.FNN([2, 60, 60, 60, 60, 1], "tanh", "Glorot uniform")

# ==========================================
# 7. 模型编译与训练
# ==========================================
# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🔥 使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("⚠️  没有可用的GPU，使用CPU")

# 将模型移动到GPU
model = dde.Model(data, net)
model.compile("adam", lr=0.001)

print("🔥 开始训练：地形 + 风场驱动的火灾模拟...")
print(f"   - 山坡位置：中心 (5,5)")
print(f"   - 风向：{WIND_DIR} (右上)")
print("-" * 30)

losshistory, train_state = model.train(iterations=20000)

# 可选：L-BFGS 微调以获得更高精度
# print("进行 L-BFGS 微调...")
# model.compile("L-BFGS")
# model.train()

# ==========================================
# 8. 结果可视化
# ==========================================
print("训练完成，正在绘制结果...")

# 生成高分辨率网格
xx, yy = np.meshgrid(
    np.linspace(X_MIN, X_MAX, 300),
    np.linspace(Y_MIN, Y_MAX, 300)
)
input_points = np.vstack((xx.ravel(), yy.ravel())).T

# 预测
predicted_time = model.predict(input_points).reshape(xx.shape)

# 计算坡度场用于背景展示 (可选，增加视觉效果)
slope_bg = get_slope_field(input_points).reshape(xx.shape)

plt.figure(figsize=(12, 10))

# 1. 绘制等时线 (主图)
# 使用 'inferno' 或 'hot'  colormap 模拟火光
levels = np.linspace(0, np.max(predicted_time), 40)
contour = plt.contourf(xx, yy, predicted_time, levels=levels, cmap='inferno')
cbar = plt.colorbar(contour)
cbar.set_label('Arrival Time (s)', fontsize=12)

# 2. 叠加坡度地形轮廓 (虚线)
# 让我们看看火是不是在山腰跑得更快
terrain_contour = plt.contour(xx, yy, slope_bg, levels=10, colors='blue', linewidths=0.8, linestyles='--', alpha=0.6)
plt.clabel(terrain_contour, inline=True, fontsize=8, fmt='Slope %.1f')

# 3. 绘制风向箭头
plt.quiver(2, 8, WIND_DIR[0], WIND_DIR[1], angles='xy', scale_units='xy', scale=1, color='cyan', width=0.005, label='Wind Direction')
plt.text(2.5, 8.2, 'Wind', color='cyan', fontsize=10, fontweight='bold')

# 4. 标记火源
circle = plt.Circle((1.0, 1.0), 0.4, color='white', fill=False, linewidth=2, label='Fire Source')
plt.gca().add_patch(circle)
plt.plot(1.0, 1.0, 'w*', markersize=15)

# 标题与标签
plt.title(f'PINN Fire Spread: Terrain & Wind Effect (GPU Acceleration)\n(Hill at Center, Wind towards Top-Right)', fontsize=14)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend(loc='upper right')
plt.axis('equal')
plt.xlim(X_MIN, X_MAX)
plt.ylim(Y_MIN, Y_MAX)
plt.grid(True, linestyle=':', alpha=0.3)

# 保存
plt.savefig('../../results/fire_advanced_results/pinns_fire_terrain_wind_gpu_result.png', dpi=300, bbox_inches='tight')
print("✅ 结果已保存为 '../../results/fire_advanced_results/pinns_fire_terrain_wind_gpu_result.png'")

# 显示损失曲线
plt.figure(figsize=(8, 5))

# 每个数组包含该时间步的损失值（PDE损失和BC损失）
# 提取每个时间步的PDE损失（第一个元素）
pde_loss = [loss[0] for loss in losshistory.loss_train]
plt.plot(losshistory.steps, pde_loss, label='PDE Loss')

# 提取每个时间步的BC损失（第二个元素）
bc_loss = [loss[1] for loss in losshistory.loss_train]
plt.plot(losshistory.steps, bc_loss, label='BC Loss')
    
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss History (GPU)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig('../../results/fire_advanced_results/training_loss_gpu.png')

plt.show()