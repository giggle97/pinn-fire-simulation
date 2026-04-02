"""
DeepXDE PINN 入门测试
求解: u'' = -pi^2*sin(pi*x), x in [-1, 1]
精确解: u = sin(pi*x)

学习指南：
1. PDE定义 (第12-14行): 学习如何将数学方程转换为代码，特别是导数的计算方法
2. 几何区域定义 (第17行): 了解如何定义求解区域
3. 边界条件设置 (第20-27行): 掌握如何定义和应用边界条件
4. 数据对象创建 (第30-36行): 理解数据对象的结构和参数设置
5. 神经网络构建 (第39行): 了解神经网络的结构设计
6. 模型编译与训练 (第42-44行): 掌握模型的编译和训练过程
7. 结果可视化 (第47-67行): 了解如何评估和可视化PINN的性能
"""
import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

# 1. 定义 PDE
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_xx + np.pi ** 2 * torch.sin(np.pi * x)

# 2. 定义几何区域
geom = dde.geometry.Interval(-1, 1)

# 3. 定义边界条件
def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], -1)

def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)

bc_l = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l)
bc_r = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_r)

# 4. 数据对象
data = dde.data.PDE(
    geom, pde,
    [bc_l, bc_r],
    num_domain=256, num_boundary=2,
    solution=lambda x: np.sin(np.pi * x[:, 0:1]),
    num_test=100
)

# 5. 神经网络
net = dde.nn.FNN([1] + [64] * 3 + [1], "tanh", "Glorot normal")

# 6. 组装并训练
model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, trainstate = model.train(iterations=5000)

# 7. 可视化结果
x_test = np.linspace(-1, 1, 200)[:, None]
y_pred = model.predict(x_test)
y_exact = np.sin(np.pi * x_test)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x_test, y_exact, "b-", label="Exact")
plt.plot(x_test, y_pred, "r--", label="PINN")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.title("PINN vs Exact Solution")

plt.subplot(1, 2, 2)
plt.plot(x_test, np.abs(y_pred - y_exact), "g-")
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.title("Error")

plt.tight_layout()
plt.savefig("../results/basic_pinn_results/test_result.png", dpi=150)
print("Done! Result saved to ../results/basic_pinn_results/test_result.png")
