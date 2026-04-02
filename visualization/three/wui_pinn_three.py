import deepxde as dde
import numpy as np
import torch
import json
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
    
    物理方程: |grad(t)|^2 - (1/R(x,y))^2 = 0
    即: (dt/dx)^2 + (dt/dy)^2 = (1/R)^2
    """
    # 使用 DeepXDE 的自动微分计算梯度
    # dy_x = dt/dx, dy_y = dt/dy
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_y = dde.grad.jacobian(y, x, i=0, j=1)
    
    # --- 核心逻辑：定义非均匀速度场 R(x, y) ---
    # 假设道路在 x=4 到 x=6 之间，贯穿整个区域
    # 使用 dde.backend.where 进行向量化条件判断 (不能用 Python 原生的 if)
    
    # 判断点是否在道路区域内
    is_road = (x[:, 0:1] > 4.0) & (x[:, 0:1] < 6.0)
    
    # 根据位置赋予不同的速度 R
    # 如果是道路，R=R_ROAD，否则 R=R_FOREST
    R = torch.where(is_road, torch.tensor(R_ROAD), torch.tensor(R_FOREST))
    
    # 计算方程残差: |grad(t)|^2 - (1/R)^2
    # 目标是最小化这个残差至 0
    return dy_x**2 + dy_y**2 - (1.0 / R)**2

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
    train_distribution="pseudo" # 伪随机采样，覆盖更均匀
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
losshistory, train_state = model.train(iterations=15000)

# (可选) 如果亚当优化后效果不够好，可以用 L-BFGS 再微调一下
# model.compile("L-BFGS")
# losshistory, train_state = model.train()

# 1. 准备网格数据用于预测
x = np.linspace(0, 10, 100) # X轴 100 个点
y = np.linspace(0, 10, 100) # Y轴 100 个点
xx, yy = np.meshgrid(x, y)
points = np.vstack((xx.flatten(), yy.flatten())).T # 转换成 (N, 2) 格式

# 2. 让模型预测时间
sol = model.predict(points)

# 3. 强制时间为非负 
sol = np.maximum(0, sol)

# 4. 构建 JSON 数据结构
# 为了减少文件大小，我们只传必要的信息
data_for_js = {
    "width": 10,
    "height": 10,
    "resolution": 100, # 网格密度
    "vertices": []     # 存储 [x, y, time]
}

for i in range(len(points)):
    px, py = points[i]
    t_val = float(sol[i][0])
    
    # 过滤掉一些极端异常值可选，这里直接存入
    data_for_js["vertices"].append([round(px, 4), round(py, 4), round(t_val, 4)])

# 5. 保存为 JSON 文件
with open('../../results/visualization_results/fire_simulation_data.json', 'w') as f:
    json.dump(data_for_js, f)

print("✅ 数据已导出为 ../../results/visualization_results/fire_simulation_data.json，请在前端目录中打开。")