import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义物理参数与场景
# ==========================================
# 模拟区域大小 (0,0) 到 (10,10)
X_MIN, Y_MIN = 0, 0
X_MAX, Y_MAX = 10, 10

# 速度定义 (单位：米/秒，假设值)
R_FOREST = 2.0  # 森林：速度快
R_ROAD = 0.5    # 道路：速度慢 (阻碍火势)

# ==========================================
# 2. 定义偏微分方程 (PDE)
# ==========================================
def pde(x, y):
    """
    x: 输入坐标 [N, 2] (x, y)
    y: 网络预测的到达时间 [N, 1] (t)
    
    物理方程: |grad(t)| - 1/R(x,y) = 0
    即: sqrt((dt/dx)^2 + (dt/dy)^2) = 1/R
    """
    # 使用 DeepXDE 的自动微分计算梯度
    # dy_x = dt/dx, dy_y = dt/dy
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_y = dde.grad.jacobian(y, x, i=0, j=1)
    
    # --- 核心逻辑：定义非均匀速度场 R(x, y) ---
    # 假设道路在 x=4 到 x=6 之间，贯穿整个区域
    # 使用 torch.where 进行向量化条件判断
    
    # 判断点是否在道路区域内
    is_road = (x[:, 0:1] > 4.0) & (x[:, 0:1] < 6.0)
    
    # 根据位置赋予不同的速度 R
    # 如果是道路，R=R_ROAD，否则 R=R_FOREST
    R = torch.where(is_road, torch.tensor(R_ROAD), torch.tensor(R_FOREST))
    
    # 计算方程残差: |grad(t)| - 1/R
    # 使用平方根形式，添加小量避免除零
    grad_mag = torch.sqrt(dy_x**2 + dy_y**2 + 1e-10)
    pde_residual = grad_mag - 1.0 / R
    
    # 添加时间非负约束
    non_negative_residual = torch.relu(-y)  # 当 y < 0 时产生损失
    
    return [pde_residual, non_negative_residual]

# ==========================================
# 3. 定义几何空间与边界条件
# ==========================================
# 定义矩形区域
geom = dde.geometry.Rectangle([X_MIN, Y_MIN], [X_MAX, Y_MAX])

# 定义起火点 (初始条件)
# 假设火从左下角 (1, 1) 附近的一个小圆圈内开始烧，时刻 t=0
def boundary_fire_source(x, on_boundary):
    # 判断是否在边界上 (对于内部点源，on_boundary 可能为 False，需用几何距离判断)
    # 这里我们定义一个内部区域作为狄利克雷边界条件
    dist = np.sqrt((x[0] - 1.0)**2 + (x[1] - 1.0)**2)
    return dist < 0.5  # 半径 0.5 的圆形起火点

# 应用边界条件：在起火点范围内，时间 t = 0
ic = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_fire_source)

# ==========================================
# 4. 构建数据集
# ==========================================
# num_domain: 区域内部采样点数 (越多越精确，训练越慢)
# num_boundary: 边界采样点数
data = dde.data.PDE(
    geom, 
    pde, 
    ic, 
    num_domain=3000, 
    num_boundary=500,
    train_distribution="pseudo", # 伪随机采样，覆盖更均匀
    num_test=1000
)

# ==========================================
# 5. 定义神经网络架构
# ==========================================
# 输入层: 2 (x, y)
# 隐藏层: 4层，每层 50 个神经元 (增加深度以拟合复杂地形)
# 输出层: 1 (time t)
# 激活函数: tanh (PINN 常用)
net = dde.nn.FNN([2, 50, 50, 50, 50, 1], "tanh", "Glorot uniform")

# ==========================================
# 6. 模型编译与训练
# ==========================================
model = dde.Model(data, net)

# 使用 Adam 优化器，学习率 0.001
model.compile("adam", lr=0.001)

print("开始训练... (预计耗时 2-5 分钟，取决于硬件)")
# 训练 15000 次迭代
losshistory, train_state = model.train(iterations=30000)

# (可选) 如果亚当优化后效果不够好，可以用 L-BFGS 再微调一下
# model.compile("L-BFGS")
# losshistory, train_state = model.train()

# ==========================================
# 7. 结果可视化
# ==========================================
print("训练完成，正在生成图表...")

# 生成网格数据用于绘图
xx, yy = np.meshgrid(
    np.linspace(X_MIN, X_MAX, 200),
    np.linspace(Y_MIN, Y_MAX, 200)
)
# 将网格展平为模型输入格式 [N, 2]
input_points = np.vstack((xx.ravel(), yy.ravel())).T

# 模型预测
predicted_time = model.predict(input_points).reshape(xx.shape)

# 绘图
plt.figure(figsize=(10, 8))
# 绘制等高线 (等时线)
contour = plt.contourf(xx, yy, predicted_time, levels=50, cmap='hot')
plt.colorbar(contour, label='Arrival Time (s)')

# 绘制道路区域示意 (白色半透明条)
plt.axvspan(4.0, 6.0, color='white', alpha=0.3, label='Road (Slow Spread)')
# 标记起火点
circle = plt.Circle((1.0, 1.0), 0.5, color='blue', fill=False, linewidth=2, label='Fire Source')
plt.gca().add_patch(circle)

plt.title(f'WUI Fire Spread Simulation (PINN)\nForest Speed={R_FOREST}, Road Speed={R_ROAD}')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)

# 保存图像
plt.savefig('../../results/fire_basic_results/wui_fire_result.png', dpi=300)
print("结果已保存为 '../../results/fire_basic_results/wui_fire_result.png'")
plt.show()  # 注释掉，避免在非交互环境中阻塞

# 打印损失变化
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# 查看模型结构和参数
# print("\n===========================================")
# print("模型结构:")
# print(model.net)

# print("\n模型参数:")
# for name, param in model.net.named_parameters():
#     print(f"参数名称: {name}, 形状: {param.shape}")

# # 保存模型
# model.save("../../results/fire_basic_results/fire_spread_model")
# print("\n模型已保存为 '../../results/fire_basic_results/fire_spread_model'")

# # 保存参数
# import torch
# torch.save(model.net.state_dict(), "../../results/fire_basic_results/model_params.pt")
# print("参数已保存为 '../../results/fire_basic_results/model_params.pt'")