import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import scipy.io
from keras import models
import os
nIter = 1000
noise = 0.0
q = 100  
layers = [1, 50, 200, 500, 200,  q] 
N = 200  # 初始数据点数量 
lr=1e-3
decay_rate = 0.99  # 学习率衰减率
# 设置随机种子确保可重现性
np.random.seed(1234)
tf.random.set_seed(1234)

class PhysicsInformedNN(tf.keras.Model):
    def __init__(self, layers, lb, ub, q, dt):
        super(PhysicsInformedNN, self).__init__()
        
        self.lb = tf.constant(lb, dtype=tf.float32)
        self.ub = tf.constant(ub, dtype=tf.float32)
        self.q = max(q, 1)
        self.dt = dt
        
        # 构建神经网络
        self.nn_model = self.build_neural_network(layers)
        
        # 加载IRK权重
        irk_file = f'Butcher_IRK{q}.txt'
        if not os.path.exists(irk_file):
            raise FileNotFoundError(f"IRK weights file not found: {irk_file}. ")
            
        tmp = np.float32(np.loadtxt(irk_file, ndmin=2))
        A=tmp[:q, :]  # QxQ矩阵
        b=tmp[q, :]   # Q维向量 
        self.A = tf.constant(A, dtype=tf.float32)
        self.b = tf.constant(b, dtype=tf.float32)
    
    def build_neural_network(self, layers):
        """使用Keras API构建神经网络"""
        model = models.Sequential()
        
        # 输入归一化层
        model.add(tf.keras.layers.Lambda(lambda X: 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0))
        
        # 隐藏层
        for i in range(1, len(layers)-1):
            model.add(tf.keras.layers.Dense(layers[i], activation='tanh',
                                  kernel_initializer=tf.keras.initializers.GlorotNormal()))
        
        # 输出层 (无激活函数)
        model.add(tf.keras.layers.Dense(layers[-1], 
                              kernel_initializer=tf.keras.initializers.GlorotNormal()))
        
        return model
    
    def call(self, X):
        """前向传播"""
        return -1+(X**2-1)*self.nn_model(X)
    
    @tf.function #将该函数编译为 TensorFlow 图（graph），提升运行效率。
    def net_U0(self, x): #输入的空间坐标，预期是一个 NumPy 数组或 Tensor，形状为 (N, 1)，其中 N 是空间采样点数量。
        # 第一次前向累积：用于计算一阶导数 U_x
        with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as acc1:
            # 第二次嵌套前向累积：用于计算二阶导数 U_xx
            with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as acc2:
                U = self.call(x)      # (N, q)
                # 一阶导数: dU/dx
                U_x = acc2.jvp(U)              # (N, q)
            # 二阶导数: d²U/dx² = d(U_x)/dx
            U_xx = acc1.jvp(U_x)               # (N, q)
        # Allen-Cahn 方程残差
        F = 5.0 * U - 5.0 * tf.math.pow(U, 3) + 0.0005 * U_xx  # (N, q)

        # 应用 IRK 时间积分格式
         
        U0 = U + self.dt * tf.linalg.matmul(F, tf.transpose(self.A))# (N, q)
        weighted_sum = tf.reduce_sum(F * self.b, axis=1, keepdims=True)  # (N, 1)
        U1 = U0 + self.dt * weighted_sum
        return U0, U1

    def train_step(self,x0,u0, u1, optimizer):
        """单个训练步骤"""
        u0 = tf.convert_to_tensor(u0, dtype=tf.float32)
        u1 = tf.convert_to_tensor(u1, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # 计算初始时刻的预测值 (只使用第一列作为初始解)
            U0, U1 = self.net_U0(x0)     #千万不要放在tf.GradientTape外面，否则梯度计算会出错
            loss_u0 = tf.reduce_mean(tf.square(u0 - U0)) #广播
            loss_u1 = tf.reduce_mean(tf.square(u1 - U1))
            total_loss = loss_u0 + loss_u1
        
        # 计算梯度并更新权重
        gradients = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return total_loss.numpy(), loss_u0.numpy(), loss_u1.numpy()
    
    def predict(self, x_star):
        """预测给定空间点的解"""
        x_star = tf.convert_to_tensor(x_star, dtype=tf.float32)
        U0_star, U1_star = self.net_U0(x_star)
        U0_star = tf.reduce_mean(U0_star, axis=1, keepdims=True)
        U1_star = tf.reduce_mean(U1_star, axis=1, keepdims=True)
        return U0_star.numpy(),U1_star.numpy()

def main():
    # 参数设置
    lb = np.array([-1.0])
    ub = np.array([1.0])
    
    # 检查数据文件是否存在
    data_file = 'AC.mat'
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}. ")
    
    # 加载数据
    data = scipy.io.loadmat(data_file)
    t = data['tt'].flatten()[:, None]  # T x 1
    x = data['x'].flatten()[:, None]   # N x 1
    Exact = np.real(data['uu']).T       # T x N
    
    # 选择时间点
    idx_t0 = 20
    idx_t1 = 180
    dt = t[idx_t1] - t[idx_t0]
    print(f"Time step: dt = {dt[0]:.4f}, from t={t[idx_t0][0]:.4f} to t={t[idx_t1][0]:.4f}")
    
    # 准备初始数据
    idx_x = np.random.choice(Exact.shape[1], N, replace=False) 
    x0 = x[idx_x, :]
    u0 = Exact[idx_t0, idx_x][:, None]  # 确保形状为 (N, 1)
    u0 = u0 + noise * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])
    u1 = Exact[idx_t1, idx_x][:, None]  # 确保形状为 (N, 1)
    u1 = u1 + noise * np.std(u1) * np.random.randn(u1.shape[0], u1.shape[1])
    
    # 测试数据
    x_star = x
    # 创建模型
    model = PhysicsInformedNN(layers, lb, ub, q, dt)
    
    # 优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    scheduler= tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_rate=decay_rate,
        decay_steps=nIter
    )
    # 训练模型

    start_time = time.time()
    
    print("Starting training...")
    for it in range(nIter):
        loss_total, loss_u0, loss_u1 = model.train_step(tf.convert_to_tensor(x0, dtype=tf.float32), tf.convert_to_tensor(u0, dtype=tf.float32), tf.convert_to_tensor(u1, dtype=tf.float32), optimizer)
        current_lr=scheduler(optimizer.iterations)
        optimizer.learning_rate=current_lr
        if it % 10 == 0:
            elapsed = time.time() - start_time
            print(f'Iteration: {it:5d}, Loss: {loss_total:.3e}, '
                  f'u0 Loss: {loss_u0:.3e}, '
                  f'u1 Loss: {loss_u1:.3e}, '
                  f'Time: {elapsed:.2f}s')
            start_time = time.time()
    
    # 预测
    U0_pred, U1_pred = model.predict(x_star)
    u0_exact = Exact[idx_t0, :]  # (N,)
    error_u0 = np.linalg.norm(U0_pred - u0_exact, 2) / np.linalg.norm(u0_exact, 2)
    print(f'\nRelative L2 error at t={t[idx_t0][0]:.4f}: {error_u0:.3e}')
    u1_exact = Exact[idx_t1, :]  # (N,)
    error_u1 = np.linalg.norm(U1_pred - u1_exact, 2) / np.linalg.norm(u1_exact, 2)
    print(f'\nRelative L2 error at t={t[idx_t1][0]:.4f}: {error_u1:.3e}')
    
    # 绘制结果
    plot_results(t, x, Exact, idx_t0, idx_t1, x_star, U0_pred, U1_pred)


def plot_results(t, x, Exact, idx_t0, idx_t1, x_star, U0_pred, U1_pred):
    """绘制结果 - 按要求修改：t0时刻也显示预测曲线而非训练点"""
    plt.figure(figsize=(14, 10))
    
    # 创建网格图
    gs = plt.GridSpec(2, 2, height_ratios=[3, 2.5])
    
    # 图1: 精确解的热图
    ax1 = plt.subplot(gs[0, :])
    h = ax1.imshow(Exact.T, interpolation='nearest', cmap='jet', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(h, ax=ax1, label='u(t,x)')
    
    # 标记t0和t1的位置
    ax1.plot([t[idx_t0], t[idx_t0]], [x.min(), x.max()], 'w--', linewidth=1.5, label=f't_0 = {t[idx_t0][0]:.2f}')
    ax1.plot([t[idx_t1], t[idx_t1]], [x.min(), x.max()], 'w-', linewidth=1.5, label=f't_1 = {t[idx_t1][0]:.2f}')
    
    ax1.set_xlabel('t', fontsize=14)
    ax1.set_ylabel('x', fontsize=14)
    ax1.set_title('Exact Solution u(t,x)', fontsize=16)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 图2: t0时刻的解 - 按要求修改为预测曲线 vs 精确解
    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(x, Exact[idx_t0, :], 'b-', linewidth=2.5, label='Exact Solution')
    ax2.plot(x_star, U0_pred, 'r--', linewidth=2.5, label='PINN Prediction')
    
    # 计算相对误差
    error_u0 = np.linalg.norm(U0_pred - Exact[idx_t0, :], 2) / np.linalg.norm(Exact[idx_t0, :], 2)
    
    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('u(t,x)', fontsize=14)
    ax2.set_title(f'Solution at t = {t[idx_t0][0]:.2f}\nRelative Error: {error_u0:.3e}', fontsize=14)
    ax2.set_xlim([x.min()-0.1, x.max()+0.1])
    ax2.set_ylim([-1.1, 1.1])
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 图3: t1时刻的预测与精确解比较
    ax3 = plt.subplot(gs[1, 1])
    ax3.plot(x, Exact[idx_t1, :], 'b-', linewidth=2.5, label='Exact Solution')
    ax3.plot(x_star, U1_pred, 'r--', linewidth=2.5, label='PINN Prediction')
    
    # 计算相对误差
    error_u1 = np.linalg.norm(U1_pred - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
    
    ax3.set_xlabel('x', fontsize=14)
    ax3.set_ylabel('u(t,x)', fontsize=14)
    ax3.set_title(f'Solution at t = {t[idx_t1][0]:.2f}\nRelative Error: {error_u1:.3e}', fontsize=14)
    ax3.set_xlim([x.min()-0.1, x.max()+0.1])
    ax3.set_ylim([-1.1, 1.1])
    ax3.legend(loc='best', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('AC_results.png', dpi=300, bbox_inches='tight')
    print("Results saved to 'AC_results.png'")
    plt.show()


if __name__ == "__main__":
    main()