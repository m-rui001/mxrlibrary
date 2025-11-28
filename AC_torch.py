import numpy as np  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import matplotlib.pyplot as plt  
import time  
import scipy.io  
import os  
from torch.func import functional_call, vmap, jacfwd  
  
# 设置随机种子  
np.random.seed(1234)  
torch.manual_seed(1234)  
  
# 配置参数  
Iter = 1000  
q = 100  
layers = [1, 20, 50, 200, 500, 200, q]  
N = 200  
Vnumber = 2000  
lr = 1e-3  
decay_rate = 0.995  
  
# 选择设备  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(f"Using device: {device}")  
  
class PhysicsInformedNN(nn.Module):  
    def __init__(self, layers, q, dt):  
        super(PhysicsInformedNN, self).__init__()  
        self.q = max(1, q)  
        self.dt = torch.tensor(dt, dtype=torch.float32, device=device)  
          
        self.nn_model = self.build_model(layers) 

          
        irk_file = f'Butcher_IRK{q}.txt'  
        if not os.path.exists(irk_file):  
            raise FileNotFoundError(f"IRK file {irk_file} not found")  
              
        tmp = np.float32(np.loadtxt(irk_file, ndmin=2))  
        self.register_buffer('A', torch.tensor(tmp[:q, :], dtype=torch.float32))  
        self.register_buffer('b', torch.tensor(tmp[-1:, :], dtype=torch.float32))  
  
    def build_model(self, layers):  
        modules = []  
        for i in range(1, len(layers) - 1):  
            layer = nn.Linear(layers[i-1], layers[i])  
            nn.init.xavier_normal_(layer.weight)  
            nn.init.zeros_(layer.bias)  
            modules.append(layer)  
            modules.append(nn.Tanh())  
          
        last_layer = nn.Linear(layers[-2], layers[-1])  
        nn.init.xavier_normal_(last_layer.weight)  
        nn.init.zeros_(last_layer.bias)  
        modules.append(last_layer)  
        return nn.Sequential(*modules)  
  
    # 这是一个纯函数版本的 forward，用于 torch.func  
    def functional_forward(self, params, buffers, x):  
        # 模拟 input_transform: 2.0*(x+1.0)/2.0 - 1.0 = x  
        # 如果你有其他归一化逻辑，写在这里  
        x_trans = x   
        # 使用 functional_call 调用 nn_model  
        output = functional_call(self.nn_model, (params, buffers), (x_trans,))  
        # 硬边界条件  
        return -1.0 + (x**2 - 1.0) * output  
  
    def net_U0(self, x):  
        """  
        使用 torch.func 极速计算 Jacobian 和 Hessian  
        """  
        # 提取模型当前的参数和缓冲区  
        params = dict(self.nn_model.named_parameters())  
        buffers = dict(self.nn_model.named_buffers())  
  
        # 定义一个计算单个样本 (1,) -> (q,) 及其导数的函数  
        def compute_per_sample(x_sample):  
            # x_sample shape: (1,)  
              
            # 1. 定义只有 x 作为输入的纯函数  
            def f(x_in):  
                return self.functional_forward(params, buffers, x_in)  
              
            # 2. 前向计算值 U (q,)  
            u_val = f(x_sample)  
              
            # 3. 计算一阶导数 Jacobian  
            # jacfwd (前向模式) 对于 输入维度(1) < 输出维度(100) 是最快的  
            # 结果 shape: (q, 1)  
            du_dx = jacfwd(f)(x_sample)  
              
            # 4. 计算二阶导数 Hessian (Jacobian of Jacobian)  
            # 结果 shape: (q, 1, 1)  
            d2u_dx2 = jacfwd(jacfwd(f))(x_sample)  
              
            return u_val, du_dx, d2u_dx2  
  
        # 使用 vmap 将单样本函数向量化，并行处理整个 batch N  
        # x shape: (N, 1) -> vmap 处理 dim 0  
        U, U_x, U_xx = vmap(compute_per_sample)(x)  
  
        # 调整形状:   
        # U: (N, q) - 已经是这个形状  
        # U_x: (N, q, 1) -> squeeze -> (N, q)  
        # U_xx: (N, q, 1, 1) -> squeeze -> (N, q)  
        U_x = U_x.squeeze(-1)  
        U_xx = U_xx.squeeze(-1).squeeze(-1)  
          
        # === 物理方程计算 (保持不变) ===  
        # F = -1.0 * (5.0 * U - 5.0 * U^3 + 0.0005 * U_xx)  
        F = -1.0 * (5.0 * U - 5.0 * torch.pow(U, 3) + 0.0005 * U_xx)  
          
        difference = F @ self.b.t()  
        U0 = U + self.dt * (F @ self.A.t())  
        U1 = U0 - self.dt * difference  
          
        return U0, U1  
  
    def predict(self, x_star):  
        self.eval()  
        # 预测时不需要求导，直接用 forward 或者 net_U0 均可  
        # 为了方便还是用 net_U0，虽然多算了导数但数据量小无所谓  
        x_star_tensor = torch.tensor(x_star, dtype=torch.float32, device=device)  
        # x_star_tensor.requires_grad = False # 预测不需要梯度追踪  
          
        # 但我们的 net_U0 内部依赖 vmap，不需要 requires_grad  
        # 注意：vmap 版本不需要 requires_grad=True，它通过函数变换处理  
        with torch.no_grad():  
            U0_star, U1_star = self.net_U0(x_star_tensor)  
          
        U0_pred = torch.mean(U0_star, dim=1)  
        U1_pred = torch.mean(U1_star, dim=1)  
        return U0_pred.cpu().numpy(), U1_pred.cpu().numpy()  
  
def train_step(model, x0, u0, u1, Vpoint, optimizer):  
    model.train()  
    optimizer.zero_grad()  
      
    # 使用 torch.func 计算的输出仍然连接着 model.parameters 的计算图  
    # 因此 loss.backward() 可以正常更新权重  
    U0, U1 = model.net_U0(x0)  
    U0_V, U1_V = model.net_U0(Vpoint)  
      
    loss_u0 = torch.mean((u0 - U0) ** 2)  
    loss_u1 = torch.mean((u1 - U1) ** 2)  
      
    # 方差损失  
    loss_v = torch.mean(torch.var(U0_V, dim=1, unbiased=False)) + torch.mean(torch.var(U1_V, dim=1, unbiased=False))  
      
    loss = loss_u0 + loss_u1 + loss_v  
      
    loss.backward()  
    optimizer.step()  
      
    return loss.item(), loss_u0.item(), loss_u1.item()  
  
def plot_results(t, x, Exact, idx_t0, idx_t1, x_star, U0_pred, U1_pred, iters, losses, losses_u0, losses_u1):  
    plt.figure(figsize=(14, 10))  
    gs = plt.GridSpec(2, 2, height_ratios=[3, 2.5])  
      
    ax1 = plt.subplot(gs[0, :])  
    ax1.plot(iters, losses, 'k-', linewidth=2, label='Total Loss')  
    ax1.plot(iters, losses_u0, 'b--', linewidth=1.5, label='Loss_u0')  
    ax1.plot(iters, losses_u1, 'r--', linewidth=1.5, label='Loss_u1')  
    ax1.set_yscale('log')  
    ax1.set_xlabel('Iteration')  
    ax1.set_ylabel('Loss')  
    ax1.set_title('Training Loss Curves')  
    ax1.legend()  
    ax1.grid(True, which="both", linestyle='--', alpha=0.7)  
  
    ax2 = plt.subplot(gs[1, 0])  
    ax2.plot(x, Exact[idx_t0, :], 'b-', linewidth=2.5, label='Exact')  
    ax2.plot(x_star, U0_pred, 'r--', linewidth=2.5, label='PINN')  
    error_u0 = np.linalg.norm(U0_pred - Exact[idx_t0, :], 2) / np.linalg.norm(Exact[idx_t0, :], 2)  
    ax2.set_title(f't0 Error: {error_u0:.3e}')  
    ax2.legend()  
    ax2.grid(True)  
  
    ax3 = plt.subplot(gs[1, 1])  
    ax3.plot(x, Exact[idx_t1, :], 'b-', linewidth=2.5, label='Exact')  
    ax3.plot(x_star, U1_pred, 'r--', linewidth=2.5, label='PINN')  
    error_u1 = np.linalg.norm(U1_pred - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)  
    ax3.set_title(f't1 Error: {error_u1:.3e}')  
    ax3.legend()  
    ax3.grid(True)  
  
    plt.tight_layout()  
    plt.savefig('AC_results_torch_optimized.png', dpi=300)  
    # plt.show()   
  
def main():  
    data_file = 'AC.mat'  
    if not os.path.exists(data_file):  
        print(f"Error: {data_file} not found.")  
        return  
          
    data = scipy.io.loadmat(data_file)  
    t = data['tt'].reshape(-1, 1)  
    x = data['x'].reshape(-1, 1)  
    Exact = np.real(data['uu']).T  
  
    idx_t0 = 20  
    idx_t1 = 180  
    dt = t[idx_t1] - t[idx_t0]  
    print(f'Time step dt={dt[0]:.4f}')  
      
    idx_x = np.random.choice(x.shape[0], N, replace=False)  
    x_train = x[idx_x, :]  
    u0 = Exact[idx_t0, idx_x].reshape(-1, 1)  
    u1 = Exact[idx_t1, idx_x].reshape(-1, 1)  
    Vpoint_np = np.random.uniform(-1.0, 1.0, (Vnumber, 1))  
  
    x_train_tf = torch.tensor(x_train, dtype=torch.float32, device=device)  
    u0_tf = torch.tensor(u0, dtype=torch.float32, device=device)  
    u1_tf = torch.tensor(u1, dtype=torch.float32, device=device)  
    Vpoint_tf = torch.tensor(Vpoint_np, dtype=torch.float32, device=device)  
  
    model = PhysicsInformedNN(layers, q, dt).to(device) 
    model=torch.compile(model, mode="max-autotune")

    optimizer = optim.Adam(model.parameters(), lr=lr)  
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)  
      
    start_time = time.time()  
    print("Start training (Optimized)")  
  
    losses, losses_u0, losses_u1, iters_list = [], [], [], []  
  
    for it in range(Iter):  
        loss, l_u0, l_u1 = train_step(model, x_train_tf, u0_tf, u1_tf, Vpoint_tf, optimizer)  
          
        if (it + 1) % 10 == 0:  
            scheduler.step()  
  
        losses.append(loss)  
        losses_u0.append(l_u0)  
        losses_u1.append(l_u1)  
        iters_list.append(it)  
  
        if it % 10 == 0:   
            elapsed = time.time() - start_time  
            print(f'Iter: {it}, Loss: {loss:.3e}, u0: {l_u0:.3e}, u1: {l_u1:.3e}, Time: {elapsed:.2f}s')  
            start_time = time.time()  
  
    U0_pred, U1_pred = model.predict(x)  
    error_u0 = np.linalg.norm(U0_pred - Exact[idx_t0, :].flatten(), 2) / np.linalg.norm(Exact[idx_t0, :].flatten(), 2)  
    error_u1 = np.linalg.norm(U1_pred - Exact[idx_t1, :].flatten(), 2) / np.linalg.norm(Exact[idx_t1, :].flatten(), 2)  
    print(f'Error t0: {error_u0:.3e}, Error t1: {error_u1:.3e}')  
      
    plot_results(t, x, Exact, idx_t0, idx_t1, x, U0_pred, U1_pred, iters_list, losses, losses_u0, losses_u1)  
  
if __name__ == "__main__":  
    main()  
