import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import qmc
import time
from thop import profile


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
set_seed(42)

# ====================== 修正后的正确Lax-Oleinik解析解求解器 ======================
class LaxOleinikSolver:
    def __init__(self):
        pass
    
    def u0(self, x):
        """初始条件"""
        return np.sin(2 * np.pi * x)
    
    def U0(self, y):
        """初值的势能函数（原函数），Lax-Oleinik的核心"""
        return (1 - np.cos(2 * np.pi * y)) / (2 * np.pi)
    
    def solve(self, x, t):
        """
        用Lax-Oleinik公式计算无粘Burgers方程的精确弱解
        x: 空间坐标数组
        t: 目标时刻（标量）
        return: 对应(x,t)的精确解u
        """
        if t < 1e-8:  # t=0直接返回初值
            return self.u0(x)
        
        u_sol = np.zeros_like(x)
        # 对每个x点，找最优源点y*
        for i, xi in enumerate(x):
            # 定义Lax-Oleinik的极小化目标函数
            def cost_function(y):
                return self.U0(y) + (xi - y) ** 2 / (2 * t)
            
            # 搜索最优y*（适配周期性，扩大搜索区间）
            res = minimize_scalar(cost_function, bounds=(-0.5, 1.5), method='bounded')
            y_star = res.x
            
            # 计算精确解
            u_sol[i] = (xi - y_star) / t
        
        return u_sol


# ====================== 网络结构 ======================
class ResidualBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.w0 = nn.Linear(h, 2 * h)
        self.w1 = nn.Linear(2 * h, h)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        residual = x
        y = self.silu(self.w0(x))
        y = self.w1(y)
        y = y + residual
        y = self.silu(y)
        return y


class TraditionalPINN(nn.Module):
    def __init__(self, h=34):
        super().__init__()
        self.h = h
        self.w_in = nn.Linear(2, h)
        self.res_blocks = nn.ModuleList([ResidualBlock(h) for _ in range(2)])
        self.w_out = nn.Linear(h, 1)
        self.silu = nn.SiLU()
    
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        y = self.silu(self.w_in(xt))
        for block in self.res_blocks:
            y = block(y)
        u = self.w_out(y)
        return u
    
    def compute_pde_loss(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # 无粘Burgers方程残差：u_t + u*u_x = 0
        pde_residual = u_t + u * u_x
        return torch.mean(pde_residual ** 2)

class TimeDependentLinear1(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w0 = nn.Linear(in_features, out_features)
        self.w1 = nn.Linear(in_features, out_features)
        # 小初始化
        nn.init.normal_(self.w1.weight, mean=0.0, std=1e-4)
    
    def forward(self, input, t):
        # 1. 计算静态部分: input @ W0.T + b0
        # 形状: (Batch, Out_Features)
        out_static = self.w0(input)
        
        # 2. 计算时间依赖部分: input @ W1.T
        # 形状: (Batch, Out_Features)
        out_dynamic = self.w1(input)
        
        # 3. 融合时间 t
        # t 的形状通常是 (Batch, 1)，直接进行广播乘法
        # 结果形状: (Batch, Out_Features)
        out = out_static + t * out_dynamic
        return out

class TimeDependentLinear2(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w0 = nn.Linear(in_features, out_features)
        self.w1 = nn.Linear(in_features, out_features)
        self.w2 = nn.Linear(in_features, out_features)
        # 小初始化
        nn.init.normal_(self.w1.weight, mean=0.0, std=1e-4)
        nn.init.normal_(self.w2.weight, mean=0.0, std=1e-4)
    def forward(self, input, x, t):

        out_static = self.w0(input)
        out_dynamic1 = self.w1(input)
        out_dynamic2 = self.w2(input)

        out = out_static + t * out_dynamic1 + x * out_dynamic2
        return out
    
class TimeDependentResidualBlock1(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.w0 = TimeDependentLinear1(h, 2 * h)
        self.w1 = TimeDependentLinear1(2 * h, h)
        self.silu = nn.SiLU()
    
    def forward(self, x, t):
        residual = x
        y = self.silu(self.w0(x, t))
        y = self.w1(y, t)
        y = y + residual
        y = self.silu(y)
        return y


class TimeDependentResidualBlock2(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.w0 = TimeDependentLinear2(h, 2 * h)
        self.w1 = TimeDependentLinear2(2 * h, h)
        self.silu = nn.SiLU()
    
    def forward(self, input, x, t):
        residual = input
        y = self.silu(self.w0(input, x, t))
        y = self.w1(y, x, t)
        y = y + residual
        y = self.silu(y)
        return y


class TINN1(nn.Module):
    def __init__(self, h=24):
        super().__init__()
        self.h = h
        self.w_in = nn.Linear(1, h)
        self.res_blocks = nn.ModuleList([TimeDependentResidualBlock1(h) for _ in range(2)])
        self.w_out = nn.Linear(h, 1)
        self.silu = nn.SiLU()
    
    def forward(self, x, t):
        y = self.silu(self.w_in(x))
        for block in self.res_blocks:
            y = block(y, t)
        u = self.w_out(y)
        return u
    
    def compute_pde_loss(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        pde_residual = u_t + u * u_x
        return torch.mean(pde_residual ** 2)
    

class TINN2(nn.Module):
    def __init__(self, h=24):
        super().__init__()
        self.h = h
        self.w_in = nn.Linear(2, h)
        self.res_blocks = nn.ModuleList([TimeDependentResidualBlock2(h) for _ in range(2)])
        self.w_out = nn.Linear(h, 1)
        self.silu = nn.SiLU()
    
    def forward(self, x, t):
        y = self.silu(self.w_in(torch.cat([x, t], dim=1)))
        for block in self.res_blocks:
            y = block(y, x, t)
        u = self.w_out(y)
        return u
    
    def compute_pde_loss(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        pde_residual = u_t + u * u_x
        return torch.mean(pde_residual ** 2)


# ====================== 训练器 ======================
class Trainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.solver = LaxOleinikSolver()
    
    def sample_initial(self, n=256):
        x = torch.rand(n, 1) * 1.0
        t = torch.zeros(n, 1)
        return x.to(self.device), t.to(self.device)
    
    def sample_boundary(self, n=128):
        # 周期边界：x=0和x=1的成对采样
        t = torch.rand(n, 1) * 0.3
        x0 = torch.zeros(n, 1)
        x1 = torch.ones(n, 1)
        return x0.to(self.device), x1.to(self.device), t.to(self.device)
    
    def sample_domain(self, n=512):
        sampler = qmc.LatinHypercube(d=2)
        samples = sampler.random(n=n)
        x = torch.tensor(samples[:, 0:1], dtype=torch.float32)
        t = torch.tensor(samples[:, 1:2], dtype=torch.float32) * 0.3
        return x.to(self.device), t.to(self.device)
    
    def compute_relative_l2_error(self, model, t_eval, n_eval=256):
        model.eval()
        errors = []
        
        with torch.no_grad():
            for ti in t_eval:
                x_np = np.linspace(0, 1, n_eval)
                x_torch = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1).to(self.device)
                t_torch = torch.full((n_eval, 1), ti, dtype=torch.float32).to(self.device)
                
                u_pred = model(x_torch, t_torch).cpu().numpy().flatten()
                u_true = self.solver.solve(x_np, ti)
                
                error = np.sqrt(np.mean((u_pred - u_true) ** 2)) / (np.sqrt(np.mean(u_true ** 2)) + 1e-8)
                errors.append(error)
        
        model.train()
        return np.mean(errors)
    
    def train_pinn(self, model, epochs=5000, lr=1e-3):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        error_history = []
        start_time = time.time()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 初值损失
            x_init, t_init = self.sample_initial(256)
            u_init_true = torch.sin(2 * torch.pi * x_init)
            u_init_pred = model(x_init, t_init)
            loss_init = torch.mean((u_init_pred - u_init_true) ** 2)
            
            # 周期边界损失：u(0,t) = u(1,t)
            x0_bc, x1_bc, t_bc = self.sample_boundary(128)
            u0_bc_pred = model(x0_bc, t_bc)
            u1_bc_pred = model(x1_bc, t_bc)
            loss_bc = torch.mean((u0_bc_pred - u1_bc_pred) ** 2)
            
            # PDE残差损失
            x_pde, t_pde = self.sample_domain(1024)
            loss_pde = model.compute_pde_loss(x_pde, t_pde)
            
            # 总损失
            loss = 1000 * loss_init + 100 * loss_bc + loss_pde
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 记录误差
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss_init: {loss_init.item():.4e}, Loss_bc: {loss_bc.item():.4e}, Loss_pde: {loss_pde.item():.4e}")
                error = self.compute_relative_l2_error(model, [0.1, 0.2, 0.3])
                error_history.append(error)
                print(f"Loss: {loss.item():.4e}, Rel L2 Error: {error:.4e}")
        
        train_time = time.time() - start_time
        return error_history, train_time


# ====================== 绘图 ======================
class Plotter:
    @staticmethod
    def plot_error_history(epochs_list, error_pinn, error_tinn1, error_tinn2=None):
        plt.figure(figsize=(10, 6))
        plt.semilogy(epochs_list, error_pinn, label='MLP', linewidth=2)
        plt.semilogy(epochs_list, error_tinn1, label='TINN1', linewidth=2)
        plt.semilogy(epochs_list, error_tinn2, label='TINN2', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Relative L2 Error', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.title('Relative L2 Error vs Training Epochs', fontsize=14)
        plt.tight_layout()
        plt.savefig('error_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_solution_comparison(pinn_model, tinn_model1, tinn_model2, solver, device):
        t_list = [0.0, 0.1, 0.2, 0.3]
        x_plot = np.linspace(0, 1, 200)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, t in enumerate(t_list):
            u_true = solver.solve(x_plot, t)
            
            x_torch = torch.tensor(x_plot, dtype=torch.float32).unsqueeze(1).to(device)
            t_torch = torch.full((200, 1), t, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                u_pinn = pinn_model(x_torch, t_torch).cpu().numpy().flatten()
                u_tinn1 = tinn_model1(x_torch, t_torch).cpu().numpy().flatten()
                u_tinn2 = tinn_model2(x_torch, t_torch).cpu().numpy().flatten()

            axes[i].plot(x_plot, u_true, 'k-', label='Analytical', linewidth=2)
            axes[i].plot(x_plot, u_pinn, 'r--', label='MLP', linewidth=1.5)
            axes[i].plot(x_plot, u_tinn1, 'b-.', label='TINN1', linewidth=1.5)
            axes[i].plot(x_plot, u_tinn2, 'g:', label='TINN2', linewidth=1.5)
            axes[i].set_xlabel('x', fontsize=12)
            axes[i].set_ylabel('u(x,t)', fontsize=12)
            axes[i].set_title(f't = {t:.1f}', fontsize=14)
            axes[i].legend(fontsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].set_ylim(-1.2, 1.2)
        
        plt.tight_layout()
        plt.savefig('solution_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


# ====================== 主函数 ======================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = Trainer(device=device)
    epochs = 10000
    # 训练模型
    set_seed(42)
    print("\n=== Training Traditional PINN ===")
    pinn = TraditionalPINN(h=54).to(device)
    error_pinn, time_pinn = trainer.train_pinn(pinn, epochs=epochs)
    
    set_seed(42)
    print("\n=== Training TINN1 ===")
    tinn1 = TINN1(h=38).to(device)
    error_tinn1, time_tinn1 = trainer.train_pinn(tinn1, epochs=epochs)
    
    set_seed(42)
    print("\n=== Training TINN2 ===")
    tinn2 = TINN2(h=31).to(device)
    error_tinn2, time_tinn2 = trainer.train_pinn(tinn2, epochs=epochs)
    
    # 输出训练时间
    print(f"\n=== Training Time Summary ===")
    print(f"MLP: {time_pinn:.2f} seconds")
    print(f"TINN1: {time_tinn1:.2f} seconds")
    print(f"TINN2: {time_tinn2:.2f} seconds")

    # 绘图
    if len(error_pinn) > 0 :
        epochs_list = np.arange(500, epochs + 1, 500)
    else:
        epochs_list = []
    plotter = Plotter()
    plotter.plot_error_history(epochs_list, error_pinn, error_tinn1, error_tinn2)
    plotter.plot_solution_comparison(pinn, tinn1, tinn2, trainer.solver, device)

def calculate_metrics():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    set_seed(42)
    pinn = TraditionalPINN(h=54).to(device)
    
    set_seed(42)
    tinn1 = TINN1(h=38).to(device)
    
    set_seed(42)
    tinn2 = TINN2(h=31).to(device)

    print("\n" + "="*50)
    print("模型参数量和计算量分析")
    print("="*50)

    # 准备输入数据
    batch_size = 1
    x = torch.randn(batch_size, 1).to(device)
    t = torch.randn(batch_size, 1).to(device)

    models_to_evaluate = [
        ("Traditional PINN (MLP)", pinn),
        ("TINN1", tinn1),
        ("TINN2", tinn2)
    ]
    for name, model in models_to_evaluate:
        model.eval()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        try:
            macs, _ = profile(model, inputs=(x, t), verbose=False)
            flops = macs * 2
            
            print(f"\n{name}:")
            print(f"  参数量 (Raw): {params:,}")
            print(f"  MACs (Raw):   {macs:,}")
            print(f"  FLOPs (Raw):  {flops:,}")

        except Exception as e:
            print(f"{name} 计算失败: {e}")
main()