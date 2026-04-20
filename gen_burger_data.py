import numpy as np
from scipy.fft import ifft, fft, fftfreq
import matplotlib.pyplot as plt
import pickle

# ==========================================
# 1. 配置模块
# ==========================================
class Config:
    """集中管理所有物理和数值参数"""
    # 空间与网格参数
    N = 2000           # 样本数
    Nx = 128           # 求解空间网格点数
    L = 1.0            # 空间域长度
    
    # 初始场频谱参数
    sigma = 25
    tau = 5
    gamma = 4
    
    # 物理与时间参数
    nu = 0.01          # 粘性系数
    T = 1.0            # 求解总时间
    Nt = 128           # 输出的时间步数
    
    # 数值参数
    cfl_safety = 0.05   # CFL 安全系数
    time_steps = None

    @staticmethod
    def source_func(x, t):
        return 0

class SolverWorkspace:
    """预分配求解器工作内存"""
    def __init__(self, cfg):
        # 2/3 规则去混叠掩码
        k_idx = fftfreq(cfg.Nx, d=1.0/cfg.Nx)
        self.dealias_mask = (np.abs(k_idx) <= cfg.Nx // 3).astype(np.float64)
        # 预分配 RK4 临时数组
        shape = (cfg.N, cfg.Nx)
        self.k1 = np.empty(shape, dtype=np.float64)
        self.k2 = np.empty(shape, dtype=np.float64)
        self.k3 = np.empty(shape, dtype=np.float64)
        self.k4 = np.empty(shape, dtype=np.float64)
        self.u_temp = np.empty(shape, dtype=np.float64)

# ==========================================
# 2. 核心逻辑模块
# ==========================================
def generate_initial_conditions(cfg: Config):
    """直接在求解网格分辨率下生成初始条件"""
    dx = cfg.L / cfg.Nx
    # 直接使用求解网格的波数
    k = fftfreq(cfg.Nx, d=dx)
    
    # 计算功率谱
    S_k = cfg.sigma**2 * (cfg.tau**2 + (2 * np.pi * k)**2)**(-cfg.gamma)
    
    # 生成随机复数
    z_complex = (np.random.randn(cfg.N, cfg.Nx) + 
                 1j * np.random.randn(cfg.N, cfg.Nx)) / np.sqrt(2)
    U = z_complex * np.sqrt(S_k)
    
    # 保证共轭对称以获得实数场
    U[:, cfg.Nx//2+1:] = np.conj(U[:, cfg.Nx//2-1:0:-1])
    U[:, cfg.Nx//2] = U[:, cfg.Nx//2].real
    U[:, 0] = U[:, 0].real
    # 直接 IFFT 得到初始场
    u0 = ifft(U, axis=1).real * cfg.Nx
    
    x = np.linspace(0, cfg.L, cfg.Nx, endpoint=False)
    return x, u0
class SolverWorkspace:
    """预分配求解器工作内存及 ETDRK4 系数"""
    def __init__(self, cfg, k_solve, dt_sub):
        # 2/3 规则去混叠掩码
        k_idx = fftfreq(cfg.Nx, d=1.0/cfg.Nx)
        self.dealias_mask = (np.abs(k_idx) <= cfg.Nx // 3).astype(np.float64).reshape(1, -1)
        
        # --- 计算 ETDRK4 系数 (Kassam & Trefethen 2005) ---
        L_op = -cfg.nu * k_solve**2  # 线性算子
        
        # 复轮廓线积分点，避免大数吃小数
        M_contour = 32
        r = np.exp(1j * np.pi * (np.arange(1, M_contour + 1) - 0.5) / M_contour)
        
        # 广播计算 LR: shape (Nx, M_contour)
        LR = dt_sub * L_op[:, None] + r[None, :]
        
        self.E = np.exp(dt_sub * L_op).reshape(1, -1)
        self.E2 = np.exp(dt_sub * L_op / 2.0).reshape(1, -1)
        
        # 使用轮廓线积分计算系数
        Q2 = dt_sub * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
        f1 = dt_sub * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1))
        f2 = dt_sub * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1))
        f3 = dt_sub * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1))
        
        self.Q2 = Q2.reshape(1, -1)
        self.f1 = f1.reshape(1, -1)
        self.f2 = f2.reshape(1, -1)
        self.f3 = f3.reshape(1, -1)
        
        # 预分配 ETDRK4 临时数组 (复数域)
        shape = (cfg.N, cfg.Nx)
        self.Nv = np.empty(shape, dtype=np.complex128)
        self.Na = np.empty(shape, dtype=np.complex128)
        self.Nb = np.empty(shape, dtype=np.complex128)
        self.Nc = np.empty(shape, dtype=np.complex128)
        self.a = np.empty(shape, dtype=np.complex128)
        self.b = np.empty(shape, dtype=np.complex128)
        self.c = np.empty(shape, dtype=np.complex128)

def compute_nonlinear(u_hat, k, dealias_mask, x, t, source_func):
    """仅计算非线性项和源项的傅里叶系数 (去除了线性项)"""
    # 去混叠
    u_hat_d = u_hat * dealias_mask
    
    # 变换到物理空间计算非线性乘积
    u = ifft(u_hat_d, axis=1).real
    u_x = ifft(1j * k * u_hat_d, axis=1).real
    
    # 源项
    S = source_func(x, t)
    
    # 变换回谱空间
    return fft(-u * u_x + S, axis=1)

def compute_dt(dx, u, cfg):
    if cfg.time_steps is None:
        max_u = np.max(np.abs(u))
        dt_safe = cfg.cfl_safety * dx / max_u
        n_substeps = max(1, int(np.ceil((cfg.T / (cfg.Nt - 1)) / dt_safe)))
        dt_sub = (cfg.T / (cfg.Nt - 1)) / n_substeps
    else:
        n_substeps = cfg.time_steps
        dt_sub = (cfg.T / (cfg.Nt - 1)) / n_substeps
    return dt_sub, n_substeps

def solve_burgers(cfg: Config, u0: np.ndarray, x: np.ndarray):
    """使用伪谱法和 ETDRK4 求解 Burgers 方程"""
    dx = x[1] - x[0]
    k_solve = 2 * np.pi * fftfreq(cfg.Nx, d=dx)
    t_out = np.linspace(0, cfg.T, cfg.Nt)
    
    # 计算时间步长 (ETDRK4 只受对流CFL限制，dt正比于dx)
    dt_sub, n_substeps = compute_dt(dx, u0, cfg)
    # 初始化工作区 (包含ETDRK4系数)
    ws = SolverWorkspace(cfg, k_solve, dt_sub)
    
    U_sol = np.zeros((cfg.N, cfg.Nt, cfg.Nx))
    U_sol[:, 0, :] = u0
    
    # 初始场变换到傅里叶空间
    u_hat = fft(u0, axis=1)
    
    print(f"开始求解 Burgers 方程 (ETDRK4)...")
    print(f"空间步长 dx={dx:.4f}")
    print(f"每输出步内部子步数: {n_substeps}, 子步长 dt={dt_sub:.6f}")
        
    for n in range(1, cfg.Nt):
        t_start = t_out[n-1]
        for s in range(n_substeps):
            t_sub = t_start + s * dt_sub
            
            # ETDRK4 步进 (在傅里叶空间进行)
            ws.Nv[:] = compute_nonlinear(u_hat, k_solve, ws.dealias_mask, x, t_sub, cfg.source_func)
            np.add(ws.E2 * u_hat, ws.Q2 * ws.Nv, out=ws.a)
            
            ws.Na[:] = compute_nonlinear(ws.a, k_solve, ws.dealias_mask, x, t_sub + 0.5*dt_sub, cfg.source_func)
            np.add(ws.E2 * u_hat, ws.Q2 * ws.Na, out=ws.b)
            
            ws.Nb[:] = compute_nonlinear(ws.b, k_solve, ws.dealias_mask, x, t_sub + 0.5*dt_sub, cfg.source_func)
            np.add(ws.E2 * ws.a, ws.Q2 * (2*ws.Nb - ws.Nv), out=ws.c)
            
            ws.Nc[:] = compute_nonlinear(ws.c, k_solve, ws.dealias_mask, x, t_sub + dt_sub, cfg.source_func)
            
            u_hat[:] = ws.E * u_hat + ws.f1 * ws.Nv + 2 * ws.f2 * (ws.Na + ws.Nb) + ws.f3 * ws.Nc
            
        # 反变换回物理空间保存
        U_sol[:, n, :] = ifft(u_hat, axis=1).real
        if n % 20 == 0:
            print(f"已完成 t = {t_out[n]:.2f}")

        dt_sub, n_substeps = compute_dt(dx, U_sol[:, n, :], cfg)
        ws = SolverWorkspace(cfg, k_solve, dt_sub)
        print(f"更新子步长: dt={dt_sub:.6f}, n_substeps={n_substeps}")
    print("求解完成")
    return U_sol, t_out

# ==========================================
# 3. 可视化与IO模块
# ==========================================
def plot_sample_solution(x, U_sol, t_out, sample_idx=0, nu=0.01):
    """绘制单个样本的时间演化图"""
    U_plot = U_sol[sample_idx]
    plt.figure(figsize=(10, 6))
    
    # 智能选择要绘制的时刻索引
    time_indices = [0, len(t_out)//16, len(t_out)//8, len(t_out)//4, len(t_out)//2, -1]
    
    for t_idx in time_indices:
        plt.plot(x, U_plot[t_idx, :], label=f't = {t_out[t_idx]:.2f}')

    plt.title(f"Burgers Equation Solution (Sample {sample_idx}, $\\nu={nu}$)")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"burger_solution{sample_idx}.png", dpi=300)
    plt.close() # 模块化时建议关闭画图窗口，防止阻塞

# ==========================================
# 4. MMS 收敛阶测试模块
# ==========================================

def evaluate_order():
    p = 2.0  
    
    def exact_u(x, t):
        return np.sin(2 * np.pi * (x - t)) * np.exp(p * t)
    
    def source_func_mms(x, t, nu):
        sin_term = np.sin(2 * np.pi * (x - t))
        cos_term = np.cos(2 * np.pi * (x - t))
        exp_term = np.exp(p * t)
        exp2_term = np.exp(2 * p * t)  # e^(2pt)
        
        term1 = -2 * exp_term * np.pi * cos_term  # 时间导数部分
        term2 = exp_term * p * sin_term           # 指数增长部分
        term3 = 4 * exp_term * np.pi**2 * nu * sin_term  # 粘性扩散部分
        term4 = 2 * exp2_term * np.pi * cos_term * sin_term  # 非线性对流部分
        
        return term1 + term2 + term3 + term4


    # 3. 误差计算辅助函数
    def compute_l2_error(u_num, u_exact):
        return np.sqrt(np.mean((u_num - u_exact)**2))

    # ----------------- 空间收敛阶测试 -----------------
    print("="*50)
    print("开始测试空间收敛阶")
    Nx_list = [2, 4, 8, 16, 32]
    errors_x = []
    time_steps_fixed = 1024
    for Nx in Nx_list:
        cfg = Config()
        cfg.Nx = Nx
        cfg.N = 1
        cfg.Nt = 2
        cfg.cfl_safety = 0.01
        cfg.time_steps = time_steps_fixed
        
        x = np.linspace(0, cfg.L, Nx, endpoint=False)
        u0_exact = exact_u(x, 0)
        u0 = u0_exact.reshape(1, -1)
        
        cfg.source_func = lambda x, t: source_func_mms(x, t, cfg.nu)
        U_sol, _ = solve_burgers(cfg, u0, x)
        
        error = compute_l2_error(U_sol[0, -1, :], exact_u(x, cfg.T))
        errors_x.append(error)
        
    print(f"{'Nx':<8} {'L2 Error':<20} {'Order (p)'}")
    print("-" * 40)
    for i, Nx in enumerate(Nx_list):
        if i == 0:
            print(f"{Nx:<8} {errors_x[i]:<20.6e} {'---':<10}")
        else:
            order = np.log(errors_x[i-1] / errors_x[i]) / np.log(Nx_list[i] / Nx_list[i-1])
            print(f"{Nx:<8} {errors_x[i]:<20.6e} {order:<10.4f}")

    # ----------------- 时间收敛阶测试 -----------------
    print("\n" + "="*50)
    print("开始测试时间收敛阶")
    Nx_fixed = 256
    n_sub_list = [1024, 2048, 4096, 8192] 
    errors_t = []
    
    x = np.linspace(0, cfg.L, Nx_fixed, endpoint=False)
    u0_exact_fixed = exact_u(x, 0)
    
    for n_sub in n_sub_list:
        cfg = Config()
        cfg.Nx = Nx_fixed
        cfg.N = 1
        cfg.Nt = 2
        #通过修改 CFL 安全系数（cfg.cfl_safety），间接修改 时间步长（dt）
        max_u0 = np.max(np.abs(u0_exact_fixed))
        dx = x[1] - x[0]
        desired_dt = cfg.T / n_sub
        cfg.cfl_safety = desired_dt * max_u0 / dx
        
        u0 = u0_exact_fixed.reshape(1, -1)
        cfg.source_func = lambda x, t: source_func_mms(x, t, cfg.nu)
        
        U_sol, _ = solve_burgers(cfg, u0, x)
        error = compute_l2_error(U_sol[0, -1, :], exact_u(x, cfg.T))
        errors_t.append(error)

    print(f"{'N_sub':<8} {'dt':<15} {'L2 Error':<20} {'Order (p)'}")
    print("-" * 55)
    for i, n_sub in enumerate(n_sub_list):
        dt_val = cfg.T / n_sub
        if i == 0:
            print(f"{n_sub:<8} {dt_val:<15.6f} {errors_t[i]:<20.6e} {'---':<10}")
        else:
            order = np.log2(errors_t[i-1] / errors_t[i])
            print(f"{n_sub:<8} {dt_val:<15.6f} {errors_t[i]:<20.6e} {order:<10.4f}")

    # ----------------- 绘制收敛阶图像 -----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 空间收敛图
    ax1.loglog(Nx_list, errors_x, 'o-', label='Measured Error')
    ax1.set_xlabel("Nx (Grid Points)")
    ax1.set_ylabel("L2 Error")
    ax1.set_title("Spatial Convergence (Spectral Accuracy)")
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    
    # 时间收敛图
    ax2.loglog(n_sub_list, errors_t, 's-', label='Measured Error')
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("L2 Error")
    ax2.set_title("Temporal Convergence Order")
    ax2.grid(True, which="both", ls="--")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("mms_convergence_order.png", dpi=300)
    plt.close()
    print("\n收敛阶测试完成，图表已保存。")

def verify_burgers_residual(data_path, nu, L=1.0, sample_idx_plot=0):
    """
    读取保存的 Burgers 方程求解数据，计算方程残差并可视化分析。
    
    参数:
        data_path: str, pkl 文件路径
        nu: float, 粘性系数
        L: float, 空间域长度
        sample_idx_plot: int, 用于绘制详细时空图的样本索引
    """
    print("正在加载数据...")
    with open(data_path, "rb") as f:
        U_sol, t_out, x = pickle.load(f)
        
    N, Nt, Nx = U_sol.shape
    dx = x[1] - x[0]
    dt = t_out[1] - t_out[0]
    
    print(f"数据形状: 样本数 N={N}, 时间步 Nt={Nt}, 空间点 Nx={Nx}")
    print(f"网格参数: dx={dx:.4f}, dt={dt:.6f}, nu={nu}")
    
    # ==========================================
    # 1. 计算空间导数 (使用伪谱法，保证精度)
    # ==========================================
    k = 2 * np.pi * fftfreq(Nx, d=dx)
    
    # 对所有时间和样本做 FFT: shape (N, Nt, Nx)
    u_hat = fft(U_sol, axis=2)
    
    # u_x = IFFT(1j * k * u_hat)
    u_x = np.real(ifft(1j * k[None, None, :] * u_hat, axis=2))
    
    # u_xx = IFFT(-k^2 * u_hat)
    u_xx = np.real(ifft(-k[None, None, :]**2 * u_hat, axis=2))
    
    # ==========================================
    # 2. 计算时间导数 (二阶中心差分)
    # ==========================================
    # 舍弃第 0 步和最后一步，只在内部时间步 [1 ... Nt-2] 计算
    u_t = (U_sol[:, 2:, :] - U_sol[:, :-2, :]) / (2.0 * dt)
    
    # 截取对应的空间导数项
    u_inner = U_sol[:, 1:-1, :]
    u_x_inner = u_x[:, 1:-1, :]
    u_xx_inner = u_xx[:, 1:-1, :]
    t_inner = t_out[1:-1]
    
    # ==========================================
    # 3. 计算残差 R = u_t + u * u_x - nu * u_xx
    # ==========================================
    # 注意: 源项 S=0
    Residual = u_t + u_inner * u_x_inner - nu * u_xx_inner
    
    # ==========================================
    # 4. 误差统计与平均
    # ==========================================
    # 每个样本在 (空间 x 时间) 上的 L2 范数
    l2_error_per_sample = np.sqrt(np.mean(Residual**2, axis=(1, 2)))
    
    # 所有样本的平均 L2 误差
    avg_l2_error = np.mean(l2_error_per_sample)
    
    # 所有样本的最大绝对残差 (看极端情况)
    max_abs_residual = np.max(np.abs(Residual))
    
    # 随时间变化的平均残差 (在空间和样本上平均)
    residual_vs_time = np.sqrt(np.mean(Residual**2, axis=(0, 2)))
    
    print("\n" + "="*50)
    print("残差分析结果 (理论上越接近0越好):")
    print(f"  样本平均 L2 残差: {avg_l2_error:.6e}")
    print(f"  最大绝对残差 (点对点): {max_abs_residual:.6e}")
    print(f"  L2残差的标准差 (样本间差异): {np.std(l2_error_per_sample):.6e}")
    print("="*50)
    
    # 理论误差下界估计 (帮助理解结果)
    # 你的数据时间导数是二阶精度的，误差主要来源是 dt^2
    print(f"\n[注] 误差下界分析:")
    print(f"  时间差分引入的截断误差量级约: O(dt^2) = {dt**2:.2e}")
    print(f"  空间谱导数误差量级约: O(机器精度) = {1e-15:.2e}")
    print(f"  => 残差主要受限于你保存数据的时间间隔 dt，而非空间离散或求解器精度！\n")

    # ==========================================
    # 5. 可视化
    # ==========================================
    fig = plt.figure(figsize=(16, 10))
    
    # 图1: 残差随时间的变化 (平均)
    ax1 = plt.subplot(2, 2, 1)
    ax1.semilogy(t_inner, residual_vs_time, 'b-', linewidth=2)
    ax1.set_title("Average L2 Residual vs Time (All Samples)")
    ax1.set_xlabel("Time $t$")
    ax1.set_ylabel("L2 Residual")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # 图2: 各样本 L2 误差分布 (直方图)
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(l2_error_per_sample, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(avg_l2_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_l2_error:.2e}')
    ax2.set_title("Distribution of L2 Error Across Samples")
    ax2.set_xlabel("L2 Error")
    ax2.set_ylabel("Sample Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 单个样本的残差时空图 (由于残差极小，可能接近0，用线性色标)
    R_sample = Residual[sample_idx_plot].T
    ax3 = plt.subplot(2, 2, 3)
    # 使用 pcolormesh 画热力图
    c = ax3.pcolormesh(t_inner, x, R_sample, shading='auto', cmap='RdBu_r')
    plt.colorbar(c, ax=ax3, label='Residual $R(x,t)$')
    ax3.set_title(f"Spatiotemporal Residual Map (Sample {sample_idx_plot})")
    ax3.set_xlabel("Time $t$")
    ax3.set_ylabel("Space $x$")
    
    # 图4: 单个样本在特定时刻的残差剖面
    ax4 = plt.subplot(2, 2, 4)
    time_snap_indices = [0, len(t_inner)//4, len(t_inner)//2, -1]
    for idx in time_snap_indices:
        ax4.plot(x, R_sample[:, idx], label=f't={t_inner[idx]:.2f}')
    ax4.set_title(f"Residual Profiles at Different Times (Sample {sample_idx_plot})")
    ax4.set_xlabel("Space $x$")
    ax4.set_ylabel("Residual $R(x)$")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("burgers_residual_analysis.png", dpi=300)
    plt.show()


# ==========================================
# 5. 主执行入口
# ==========================================
if __name__ == "__main__":
    # # 1. 运行原本的随机场求解流程
    cfg = Config()
    x, u0 = generate_initial_conditions(cfg)
    U_sol, t_out = solve_burgers(cfg, u0, x)
    with open("burger_solution.pkl", "wb") as f:
        pickle.dump((U_sol, t_out, x), f)
    with open("burger_solution.pkl", "rb") as f:
        U_sol, t_out, x = pickle.load(f)
    for i in range(10):
        plot_sample_solution(x, U_sol, t_out, sample_idx=i, nu=cfg.nu)
    
    # 2. 运行 MMS 收敛阶评估
    evaluate_order()
    verify_burgers_residual(data_path="burger_solution.pkl", nu=cfg.nu, L=cfg.L, sample_idx_plot=0)