"""
Rothermel PINN v1 - 野火蔓延预测 (简化版)
=========================================
基于 Eikonal 方程: |nabla t(x,y)| = 1/R(x,y)
- t(x,y): 火线到达 (x,y) 的时间
- R(x,y): 局部蔓延速度 (Rothermel 模型)
简化假设: R 为常数 (均匀地形、均匀植被)
点火点: 区域中心
"""

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================
# 1. 物理参数
# ============================
R = 0.5  # 蔓延速度 (m/s), 简化常数

# 区域范围 (米)
X_MIN, X_MAX = 0, 100
Y_MIN, Y_MAX = 0, 100

# 点火点 (区域中心)
IGNITION = np.array([[50.0, 50.0]])


# ============================
# 2. 定义 PDE (Eikonal 方程)
# ============================
def eikonal_pde(x, y):
    """
    Eikonal 方程: sqrt((dt/dx)^2 + (dt/dy)^2) = 1/R
    残差形式: sqrt((dt/dx)^2 + (dt/dy)^2) - 1/R = 0
    """
    # DeepXDE 的自动微分
    dy_dx = dde.grad.jacobian(y, x, i=0, j=0)
    dy_dy = dde.grad.jacobian(y, x, i=0, j=1)

    grad_mag = torch.sqrt(dy_dx ** 2 + dy_dy ** 2 + 1e-10)
    return grad_mag - 1.0 / R


# ============================
# 3. 几何区域
# ============================
geom = dde.geometry.Rectangle([X_MIN, Y_MIN], [X_MAX, Y_MAX])


# ============================
# 4. 数据对象
# ============================
# 点火点条件: 用 PointSet 设置 t(ignition) = 0
bc_ignition = dde.icbc.DirichletBC(geom, lambda x: 0.0, lambda x, on: np.linalg.norm(x - IGNITION[0]) < 1.5)

data = dde.data.PDE(
    geom,
    eikonal_pde,
    [bc_ignition],
    num_domain=5000,
    num_boundary=500,
    num_test=2000
)


# ============================
# 5. 神经网络
# ============================
net = dde.nn.FNN(
    [2] + [128] * 4 + [1],
    "tanh",
    "Glorot normal"
)


# ============================
# 6. 训练
# ============================
model = dde.Model(data, net)
model.compile("adam", lr=0.001)

print("=" * 50)
print("Phase 1: Adam optimizer")
print("=" * 50)
losshistory, trainstate = model.train(iterations=10000)

print("\n" + "=" * 50)
print("Phase 2: L-BFGS fine-tuning")
print("=" * 50)
model.compile("L-BFGS")
model.train(iterations=5000)


# ============================
# 7. 可视化
# ============================
nx, ny = 200, 200
x_grid = np.linspace(X_MIN, X_MAX, nx)
y_grid = np.linspace(Y_MIN, Y_MAX, ny)
X, Y = np.meshgrid(x_grid, y_grid)
points = np.hstack([X.ravel()[:, None], Y.ravel()[:, None]])

T_pred = model.predict(points).reshape(ny, nx)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) 到达时间热力图
c1 = axes[0].contourf(X, Y, T_pred, levels=30, cmap='hot_r')
axes[0].plot(IGNITION[0, 0], IGNITION[0, 1], 'g*', markersize=15, label='Ignition')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
axes[0].set_title('Arrival Time t(x,y)')
axes[0].legend()
plt.colorbar(c1, ax=axes[0], label='Time (s)')

# (b) 等时线 (火线)
c2 = axes[1].contour(X, Y, T_pred, levels=15, colors='red', linewidths=1.5)
axes[1].clabel(c2, inline=True, fontsize=8, fmt='%.1f s')
axes[1].plot(IGNITION[0, 0], IGNITION[0, 1], 'g*', markersize=15)
axes[1].set_xlabel('x (m)')
axes[1].set_ylabel('y (m)')
axes[1].set_title('Isochrones (Firefront)')
axes[1].set_aspect('equal')

# (c) 梯度场
dy_dx_field = np.gradient(T_pred, x_grid, axis=1)
dy_dy_field = np.gradient(T_pred, y_grid, axis=0)
grad_mag = np.sqrt(dy_dx_field**2 + dy_dy_field**2)

c3 = axes[2].contourf(X, Y, grad_mag, levels=30, cmap='viridis')
axes[2].contour(X, Y, T_pred, levels=10, colors='white', linewidths=0.8, alpha=0.6)
axes[2].plot(IGNITION[0, 0], IGNITION[0, 1], 'g*', markersize=15)
axes[2].set_xlabel('x (m)')
axes[2].set_ylabel('y (m)')
axes[2].set_title('|nabla t| (should ~ 1/R = {:.1f})'.format(1/R))
plt.colorbar(c3, ax=axes[2], label='|nabla t|')

plt.tight_layout()
plt.savefig("../../results/fire_basic_results/rothermel_v1_result.png", dpi=150, bbox_inches='tight')
print("\nDone! Saved to ../../results/fire_basic_results/rothermel_v1_result.png")

# 精确解对比 (圆形扩散 t = r/R)
r = np.sqrt((X - IGNITION[0, 0])**2 + (Y - IGNITION[0, 1])**2)
t_exact = r / R
error = np.abs(T_pred - t_exact)
print(f"Mean error: {np.mean(error):.4f} s")
print(f"Max error:  {np.max(error):.4f} s")
print(f"|nabla t| mean: {np.mean(grad_mag):.4f} (theory: {1/R:.4f})")
