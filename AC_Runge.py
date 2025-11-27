import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import scipy.io
from keras import models
import os
Iter = 1000
q = 100  
layers = [1, 20, 50, 200, 500, 200,  q] 
N = 200   
Vnumber = 2000 #Number of variance points which are used to compute the variance loss(we have no imformation of real value at t0 or t1)
lr=1e-3
decay_rate = 0.995  
np.random.seed(1234)
tf.random.set_seed(1234)

class PHysicsInformedNN(tf.keras.Model):
    def __init__(self,layers,q,dt):
        super(PHysicsInformedNN,self).__init__()
        self.q=max(1,q)
        self.dt=tf.constant(dt,dtype=tf.float32)
        self.nn_model=self.build_model(layers)
        irk_file = f'Butcher_IRK{q}.txt'
        if not os.path.exists(irk_file):
            raise FileNotFoundError
        tmp=np.float32(np.loadtxt(irk_file,ndmin=2))
        self.A=tf.constant(tmp[:q,:], dtype=tf.float32)
        self.b=tf.constant(tmp[-1:,:], dtype=tf.float32)

    def build_model(self,layers):
        model=models.Sequential()
        model.add(tf.keras.layers.Lambda(lambda x:2.0*(x+1.0)/(2.0)-1.0))
        for i in range(1,len(layers)-1):
            model.add(tf.keras.layers.Dense(layers[i],activation='tanh',kernel_initializer=tf.keras.initializers.GlorotNormal()))
        model.add(tf.keras.layers.Dense(layers[-1],kernel_initializer=tf.keras.initializers.GlorotNormal()))
        return model
        
    def call(self,x):
        return -1.0+(x**2-1.0)*self.nn_model(x)
    
    @tf.function
    def net_U0(self, x):
        # x: (N, 1)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape1:
                tape1.watch(x)
                U = self.call(x)  # (N, q)
            # 计算一阶导数: dU/dx
            # batch_jacobian会计算每个样本的导数，返回形状为(N, q, 1)
            U_x = tape1.batch_jacobian(U, x)
        
        # 计算二阶导数: d²U/dx²
        U_xx_list = []
        for i in range(self.q):
            # 获取第i个分量的一阶导数 (N, 1)
            U_x_i = U_x[:, i, :]
            # 计算该分量的二阶导数 (N, 1, 1)
            U_xx_i = tape2.batch_jacobian(U_x_i, x)
            # 压缩维度为 (N,)
            U_xx_list.append(tf.squeeze(U_xx_i, axis=[1, 2]))
        
        # 堆叠所有分量的二阶导数，形成 (N, q)
        U_xx = tf.stack(U_xx_list, axis=1)
        # 压缩一阶导数的维度，形成 (N, q)
        U_x = tf.squeeze(U_x, axis=-1)
        
        # 计算F项
        F = -1.0 * (5.0 * U - 5.0 * tf.pow(U, 3) + 0.0005 * U_xx)
        
        # 计算U0和U1
        difference = tf.linalg.matmul(F, tf.transpose(self.b))
        U0 = U + self.dt * tf.linalg.matmul(F, tf.transpose(self.A))
        U1 = U0 - self.dt * difference
        
        return U0, U1

    def train_step(self,x0,u0,u1,Vpoint,optimizer):
        x0=tf.convert_to_tensor(x0,dtype=tf.float32)
        u0=tf.convert_to_tensor(u0,dtype=tf.float32)
        u1=tf.convert_to_tensor(u1,dtype=tf.float32)
        Vpoint=tf.convert_to_tensor(Vpoint,dtype=tf.float32)
        with tf.GradientTape() as tape:
            U0,U1=self.net_U0(x0)
            U0_V,U1_V=self.net_U0(Vpoint)
            loss_u0=tf.reduce_mean(tf.square(u0-U0))
            loss_u1=tf.reduce_mean(tf.square(u1-U1))
            loss_v= tf.reduce_mean(tf.math.reduce_variance(U0_V,axis=1)) + tf.reduce_mean(tf.math.reduce_variance(U1_V,axis=1))
            loss=loss_u0+loss_u1+loss_v
        gradients=tape.gradient(loss,self.trainable_variables)
        optimizer.apply_gradients(zip(gradients,self.trainable_variables))
        return loss.numpy(),loss_u0.numpy(),loss_u1.numpy()
    
    def predict(self,x_star):
        x_star=tf.constant(x_star,dtype=tf.float32)
        U0_star,U1_star=self.net_U0(x_star)
        U0_star=tf.reduce_mean(U0_star,axis=1)
        U1_star=tf.reduce_mean(U1_star,axis=1)
        return U0_star.numpy(),U1_star.numpy()
    
def main():
    data_file = 'AC.mat'
    if not os.path.exists(data_file):
        raise FileNotFoundError
    data=scipy.io.loadmat(data_file)
    t=data['tt'].reshape(-1,1)
    x=data['x'].reshape(-1,1)
    Exact=np.real(data['uu']).T

    idx_t0=20
    idx_t1=180
    dt=t[idx_t1]-t[idx_t0]
    print(f'Time step dt={dt[0]:.4f}, from t0={t[idx_t0][0]:.4f} to t1={t[idx_t1][0]:.4f}')
    idx_x=np.random.choice(x.shape[0],N,replace=False)
    x_train=x[idx_x,:]
    u0=Exact[idx_t0,idx_x].reshape(-1,1)
    u1=Exact[idx_t1,idx_x].reshape(-1,1)
    Vpoint=np.random.uniform(-1.0,1.0,(Vnumber,1))


    model=PHysicsInformedNN(layers,q,dt)
    scheduler=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,decay_rate=decay_rate,decay_steps=10)
    optimizer=tf.keras.optimizers.Adam(learning_rate=scheduler)

    start_time=time.time()
    print("Start training")

    # === 新增：用于记录 loss 的列表 ===
    losses = []
    losses_u0 = []
    losses_u1 = []
    iters = []

    for it in range(Iter):
        loss, loss_u0, loss_u1 = model.train_step(x_train, u0, u1,Vpoint, optimizer)

        # === 记录 loss ===
        losses.append(loss)
        losses_u0.append(loss_u0)
        losses_u1.append(loss_u1)
        iters.append(it)

        if it % 10 == 0:
            elapsed = time.time() - start_time
            print(f'Iterations: {it}, Loss: {loss:.3e}, Loss_u0: {loss_u0:.3e}, Loss_u1: {loss_u1:.3e}, Time: {elapsed:.2f}s')
            start_time = time.time()

    U0_pred, U1_pred = model.predict(x)
    error_u0 = np.linalg.norm(U0_pred - Exact[idx_t0, :], 2) / np.linalg.norm(Exact[idx_t0, :], 2)
    error_u1 = np.linalg.norm(U1_pred - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
    print(f'Error at t0: {error_u0:.3e}, Error at t1: {error_u1:.3e}')

    # === 修改：传入 loss 数据绘图 ===
    plot_results(t, x, Exact, idx_t0, idx_t1, x, U0_pred, U1_pred, iters, losses, losses_u0, losses_u1)

def plot_results(t, x, Exact, idx_t0, idx_t1, x_star, U0_pred, U1_pred, iters, losses, losses_u0, losses_u1):
    """绘制结果：上半部分为 loss 曲线，下半部分为 t0/t1 时刻解的对比"""
    plt.figure(figsize=(14, 10))
    
    gs = plt.GridSpec(2, 2, height_ratios=[3, 2.5])
    
    # === 图1: Loss 曲线（替代原来的热图）===
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(iters, losses, 'k-', linewidth=2, label='Total Loss')
    ax1.plot(iters, losses_u0, 'b--', linewidth=1.5, label='Loss_u0')
    ax1.plot(iters, losses_u1, 'r--', linewidth=1.5, label='Loss_u1')
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration', fontsize=14)
    ax1.set_ylabel('Loss (log scale)', fontsize=14)
    ax1.set_title('Training Loss Curves', fontsize=16)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, which="both", linestyle='--', alpha=0.7)

    # 图2: t0时刻的解
    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(x, Exact[idx_t0, :], 'b-', linewidth=2.5, label='Exact Solution')
    ax2.plot(x_star, U0_pred, 'r--', linewidth=2.5, label='PINN Prediction')
    error_u0 = np.linalg.norm(U0_pred - Exact[idx_t0, :], 2) / np.linalg.norm(Exact[idx_t0, :], 2)
    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('u(t,x)', fontsize=14)
    ax2.set_title(f'Solution at t = {t[idx_t0][0]:.2f}\nRelative Error: {error_u0:.3e}', fontsize=14)
    ax2.set_xlim([x.min()-0.1, x.max()+0.1])
    ax2.set_ylim([-1.1, 1.1])
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 图3: t1时刻的解
    ax3 = plt.subplot(gs[1, 1])
    ax3.plot(x, Exact[idx_t1, :], 'b-', linewidth=2.5, label='Exact Solution')
    ax3.plot(x_star, U1_pred, 'r--', linewidth=2.5, label='PINN Prediction')
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
