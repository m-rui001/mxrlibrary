from __future__ import annotations
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math, os, random
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import multiprocessing
import pickle

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)

class RFFBank(nn.Module):
    def __init__(self, d_in: int, d_q: int, fre_per_dim: int, fre_ub: float, device: Optional[torch.device] = None):
        super().__init__()
        self.d_in = d_in
        self.d_q = d_q
        self.fre_per_dim = fre_per_dim
        self.fre_ub = fre_ub
        self.device = device or torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"{fre_per_dim=}, {d_q=}, {d_in=}")
        # 确保 fre_per_dim 能被 d_q 整除，以便整除生成 token
        assert fre_per_dim % d_q == 0, "fre_per_dim 必须是 d_q 的整数倍"
        
        self.n_token = fre_per_dim ** d_in // d_q
        print(f"RFFBank initialized with n_token={self.n_token}, d_q={d_q}, d_in={d_in}, fre_per_dim={fre_per_dim}, fre_ub={fre_ub}")
        # 生成结构化频率矩阵 Omega
        omega, G = self._generate_structured_omega(fre_ub, fre_per_dim)
        self.register_buffer("omega", omega)
        self.register_buffer("G", G)

    def _generate_structured_omega(self, fre_up: float, fre_per_dim: int):
        """
        生成 Omega 并计算分 Token 归一化的 G
        """
        d_in = self.d_in
        d_q = self.d_q
        n_token = self.n_token
        # 1. 生成基础频率 [0, fre_up]
        freqs = torch.linspace(0, fre_up, fre_per_dim + 1, device=self.device)[1:]  # (fre_per_dim,)
        freqs = torch.cartesian_prod(*[freqs] * d_in) # (fre_per_dim^d_in, d_in)
        omega = freqs.reshape(n_token, d_q, d_in)
        G = omega[..., -1] # (n_token, d_q)
        f_min = G.min(dim=-1, keepdim=True).values # (n_token, 1)
        f_max = G.max(dim=-1, keepdim=True).values
        G = (G - f_min) / (f_max - f_min + 1e-8) * 2 - 1  # 归一化到 [-1, 1]
        omega = omega * torch.sqrt(torch.tensor(2/(d_q * n_token), device=self.device)) # 归一化 Omega 的幅度
        self.phase = torch.linspace(0, 2 * torch.pi, n_token * d_q, device=self.device).reshape(1, n_token, d_q)
        return omega, G

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = torch.einsum("nkd,bd->bnk", self.omega, x) # (B, n_token, d_q)
        cos_feat = torch.cos(proj + self.phase)
        return cos_feat

class KernelizedAttentionResBlock(nn.Module):
    def __init__(self, n_token: int, ffn_mult: int):
        super().__init__()
        self.n_token = n_token
        self.mu_proj = nn.Linear(n_token, n_token)
        self.sigma_proj = nn.Linear(n_token, n_token)
        self.ffn = nn.Sequential(
            nn.Linear(n_token, ffn_mult * n_token),
            nn.SiLU(),
            nn.Linear(ffn_mult * n_token, n_token),
        )
        self.ln_ff = nn.LayerNorm(n_token)
        self.ln_q = nn.LayerNorm(n_token)
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B = Q.shape[0]
        q = self.ln_q(Q)
        mu = F.tanh(self.mu_proj(q).reshape(B, self.n_token, 1)) # (B, n_token, 1) in [-2, 2]
        sigma = self.sigma_proj(q).reshape(B, self.n_token, 1) 
        S = torch.exp(-0.5 * ((K - mu)**2 / (sigma**2 + 1e-8)))
        A = (S * V).sum(dim=-1)  # (B, n_token)
        Q = A + Q
        Q = Q + self.ffn(self.ln_ff(Q))
        return Q
    
class MLPResBlock(nn.Module):
    def __init__(self, n_token: int, ffn_mult: int):
        super().__init__()
        self.n_token = n_token

        self.ffn = nn.Sequential(
            nn.Linear(n_token, ffn_mult * n_token),
            nn.SiLU(),
            nn.Linear(ffn_mult * n_token, n_token),
        )
        self.ln_ff = nn.LayerNorm(n_token)
    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        Q = Q + self.ffn(self.ln_ff(Q))
        return Q

class KernelizedAttentionRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_token: int,
        d_q: int,
        n_total_blocks: int,
        n_attn_blocks: int,
        ffn_mult: int,
        fre_ub: float,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        fre_per_dim =int( math.pow( n_token * d_q , 1 / d_in))
        print(f"{fre_per_dim=}, {d_q=}, {d_in=}")
        assert fre_per_dim ** d_in // d_q == n_token
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RFFBank(d_in=d_in, d_q=d_q, fre_per_dim=fre_per_dim, fre_ub=fre_ub, device=device)
        self.d_q = d_q
        self.L1 = nn.Sequential(
            nn.Linear(d_in, n_token),
            nn.SiLU(),
        )
        self.attn_blocks = nn.ModuleList([KernelizedAttentionResBlock(n_token, ffn_mult=ffn_mult) for _ in range(n_attn_blocks)])
        self.mlp_blocks = nn.ModuleList([MLPResBlock(n_token, ffn_mult=ffn_mult) for _ in range(n_total_blocks - n_attn_blocks)])
        self.readout = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = self.tokenizer(x) # (B, n_token, d_q)
        Q = self.L1(x) # (B, n_token)
        for blk in self.attn_blocks:
            Q = blk(Q, self.tokenizer.G, H)
        for blk in self.mlp_blocks:
            Q = blk(Q)
        return self.readout(Q).squeeze(-1)

torch.set_default_dtype(torch.float64)
# 辅助函数定义
def sigma_k(t, k):
    """平滑逻辑门 σ_k(t) = 1/(1+e^{-kt})"""
    return 1.0 / (1.0 + torch.exp(-k * t))

def AG(theta, a0, a1, k):
    """
    角度门控函数 (Angular Gate)
    AG(θ; a0,a1,k) = σ_k((a1-a0)/2 - |atan2(sin(θ-(a0+a1)/2), cos(θ-(a0+a1)/2))|)
    """
    mid = (a0 + a1) / 2.0
    half_width = (a1 - a0) / 2.0
    # 计算角度差并包裹到 [-π, π]
    angle_diff = torch.atan2(torch.sin(theta - mid), torch.cos(theta - mid))
    return sigma_k(half_width - torch.abs(angle_diff), k)

def BP(r, r1, r2, k):
    """
    带通函数 (Bandpass)，基于两个逻辑门的差
    BP(r; r1,r2,k) = σ_k(r-r1) * [1 - σ_k(r-r2)]
    """
    return sigma_k(r - r1, k) * (1.0 - sigma_k(r - r2, k))

def f1(x, y):
    """
    f1(x1, x2) - 包含各向异性、高频环、螺旋包络、星形间断和交叉项的复杂函数
    注意：输入 x, y 应为 PyTorch 张量，定义域 [-1, 1]
    """
    # 极坐标转换
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    
    # 角门控集合 S
    S = [
        (-0.9 * torch.pi, -0.3 * torch.pi),
        (-0.1 * torch.pi, 0.5 * torch.pi),
        (0.6 * torch.pi, 0.95 * torch.pi)
    ]
    
    # rs(θ) = 0.2 + 0.15*(θ+π)/(2π)  —— 高斯螺旋包络的中心半径
    rs = 0.2 + 0.15 * (theta + torch.pi) / (2.0 * torch.pi)
    
    # r*(θ) = 0.55 + 0.10*cos(5θ)  —— 星形间断的边界
    r_star = 0.55 + 0.10 * torch.cos(5.0 * theta)
    
    term1 = 0.0
    for a0, a1 in S:
        ag = AG(theta, a0, a1, 50)
        freq = 2.0 * torch.pi * (2.2 + 2.5 * r)
        phase = x * torch.cos(2.5 * theta) + y * torch.sin(2.5 * theta)
        term1 += ag * torch.cos(freq * phase)
    term1 *= 0.35
    bp = BP(r, 0.62, 0.78, 60)
    freq2 = 2.0 * torch.pi * (6.0 + 5.0 * r)
    phase2 = x * torch.cos(3.0 * theta) + y * torch.sin(3.0 * theta)
    term2 = 0.40 * bp * torch.cos(freq2 * phase2)
    gauss = torch.exp(-(r - rs)**2 / (2.0 * 0.04**2))
    freq3 = 2.0 * torch.pi * (3.0 + 3.0 * r)
    phase3 = x * torch.cos(theta + 0.8) + y * torch.sin(theta + 0.8)
    term3 = 0.28 * gauss * torch.cos(freq3 * phase3)
    term4 = 0.12 * torch.sign(r - r_star)
    term5 = 0.10 * torch.cos(6.0 * torch.pi * x) * torch.cos(7.0 * torch.pi * y)
    return term1 + term2 + term3 + term4 + term5

def f2(x, y):
    return torch.cos(2*torch.pi*(4+3*torch.sqrt(x**2+y**2))*(x*torch.cos(5*torch.atan2(y,x))+y*torch.sin(5*torch.atan2(y,x))))

def f3(x, y):
    """
    f3(x1, x2) = sign(sin(2π*f_x*x1) * sin(2π*f_y*x2))
    其中 f_x = f_y = 1，生成棋盘格间断模式
    """
    fx, fy = 1.0, 1.0
    return torch.sign(torch.sin(2.0 * torch.pi * fx * x) * torch.sin(2.0 * torch.pi * fy * y))

def generate_grid_data(f, n=500):
    """
    在 Omega = [-1, 1]^2 上生成 n×n 均匀网格，并计算函数 f 的值
    
    参数:
        f: 目标函数，接受两个 torch.Tensor 参数 (x, y)，返回函数值
        n: 网格分辨率（默认 500），生成 n×n 个网格点
        
    返回:
        如果 f 输出标量场:
            X: (n, n) 的网格坐标，X[i,j] = x_i
            Y: (n, n) 的网格坐标，Y[i,j] = y_j  
            Z: (n, n) 的函数值，Z = f(X, Y)
    """
    # 在 [-1, 1] 区间生成均匀分布的坐标
    with torch.no_grad():  # 关键：在此区域内不构建计算图
        x = torch.linspace(-1, 1, n)
        y = torch.linspace(-1, 1, n)
        
        # 创建二维网格 (indexing='ij' 表示矩阵坐标系，i 对应 x，j 对应 y)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # 计算函数值（利用 PyTorch 广播机制）
        Z = f(X, Y)

        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)  # (n*n, 2)
        values = Z.flatten().unsqueeze(1)  # (n*n, 1)
    return coords, values

def make_true_function(f):
    """创建一个闭包，记住要使用的函数 f"""
    def true_function(x):
        return f(x[:, 0], x[:, 1])
    return true_function
true_function = make_true_function(f1)

# =========================
# Multiscale RFF Tokenizer (scale-oriented) with cos+phase
# =========================
class MultiscaleRFFTokenizer(nn.Module):
    def __init__(
        self,
        d_in: int,
        m_base: int,
        n_scales: int,
        n_token: int,
        d_q: int,
        sigma: float,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert n_scales >= 1
        self.d_in = d_in
        self.m_base = m_base
        self.n_scales = n_scales
        self.n_token = n_token
        self.d_q = d_q
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Base frequencies ω_m ~ N(0, I / sigma^2)  (fixed buffer)
        self.register_buffer(
            "omega_base",
            torch.randn(m_base, d_in, dtype=dtype, device=device) / float(sigma),
        )
        # Build scaled frequency bank across dyadic-like scales
        scaled_omegas = []
        for k in range(self.n_scales):
            wk = self.omega_base * (2**k)
            scaled_omegas.append(wk)
        Omega = torch.cat(scaled_omegas, dim=0) # (m_base * n_scales, d_in)
        self.register_buffer("Omega", Omega)
        self.m_total = m_base * n_scales

        # Build per-scale phase segments now so we don't regenerate every forward pass.
        phase_full = 2.0 * math.pi * torch.rand(self.m_total, dtype=dtype, device=device)
        self.register_buffer("phase_full", phase_full)
        # ------------------------------------------------------------------------
        # Project grouped features to d_q (note: input width is n_token now)
        #self.proj = nn.Linear(n_token, d_q, bias=True)
        # Stabilize feature magnitude.
        # With cos+random phase, Var[cos(• + b)] = 1/2 over b, so use sqrt(2/M).
        self.scale = math.sqrt(2.0 / float(self.m_total))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_in)
        returns: tokens (B, n_tokens, d_q)
        """
        # Project x onto the frequency bank and add fixed random phases
        B = x.shape[0]
        z_raw = x @ self.Omega.t()

        phases = self.phase_full
        z = z_raw + phases  # (B, m_total)
        # Single feature per (m,k): scaled cosine with random phase
        feats = torch.cos(z) * self.scale  # (B, m_total)

        # Group per token
        tokens_in = feats.view(B, self.n_token, self.d_q)  # (B, n_token, d_q)

        # Project + normalize
        H = tokens_in  # (B, n_tokens, d_q)
        return H

# =========================
# Cross-Attention Residual Block
# =========================
class CrossAttentionResBlock(nn.Module):
    def __init__(self, d_q, n_heads, ffn_mult):
        super().__init__()
        assert d_q % n_heads == 0
        self.d_q = d_q
        self.n_heads = n_heads
        self.d_head = d_q // n_heads

        self.W_q = nn.Linear(d_q, d_q)
        self.W_k = nn.Linear(d_q, d_q)
        self.W_v = nn.Linear(d_q, d_q)
        self.W_o = nn.Linear(d_q, d_q)

        self.ffn = nn.Sequential(
            nn.Linear(d_q, ffn_mult * d_q),
            nn.SiLU(),
            nn.Linear(ffn_mult * d_q, d_q),
        )
        self.ln_q = nn.LayerNorm(d_q)
        self.ln_ff = nn.LayerNorm(d_q)
    def forward(self, Q: torch.Tensor, KV: torch.Tensor) -> torch.Tensor:
        B, L, _ = Q.shape
        q = self.ln_q(Q)
        kv = KV

        q = self.W_q(q).view(B, L, self.n_heads, self.d_head).transpose(1, 2)              # (B, H, L, Dh)
        k = self.W_k(kv).view(B, KV.shape[1], self.n_heads, self.d_head).transpose(1, 2)   # (B, H, T, Dh)
        v = self.W_v(kv).view(B, KV.shape[1], self.n_heads, self.d_head).transpose(1, 2)   # (B, H, T, Dh)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)        # (B, H, L, T)
        attn = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn, v)                                                    # (B, H, L, Dh)
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_q)            # (B, L, D)

        Q = Q + self.W_o(context)
        Q = Q + self.ffn(self.ln_ff(Q))
        return Q

class MultiscaleRFFCrossAttentionRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        ffn_mult: int,
        m_base: int,
        n_scales: int,
        n_total_blocks: int,
        n_attn_blocks: int,
        n_token: int,
        d_q: int,
        n_heads: int,
        n_queries: int,
        sigma: float,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = MultiscaleRFFTokenizer(
            d_in=d_in, m_base=m_base, n_scales=n_scales, n_token=n_token,
            d_q=d_q, sigma=sigma,
            dtype=dtype, device=device
        )
        self.d_q = d_q
        self.n_queries = n_queries
        #self.query_tokens = nn.Parameter(torch.randn(n_queries, d_q, dtype=dtype, device=device) * 0.02)

        # L1(x): input-conditioned query features (B, d_q)
        self.L1 = nn.Sequential(
            nn.Linear(d_in, n_queries * d_q),
            nn.SiLU(),
        )

        self.attn_blocks = nn.ModuleList([CrossAttentionResBlock(d_q, n_heads, ffn_mult) for _ in range(n_attn_blocks)])
        self.mlp_blocks = nn.ModuleList([MLPResBlock(d_q, ffn_mult) for _ in range(n_total_blocks - n_attn_blocks)])
        self.readout = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_q, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = self.tokenizer(x)                     # (B, n_tokens, d_q)
        l1 = self.L1(x)
        Q = l1.view(x.size(0), self.n_queries, self.d_q)

        for blk in self.attn_blocks:
            Q = blk(Q, H)
        for blk in self.mlp_blocks:
            Q = blk(Q)
        z = Q.mean(dim=1)
        return self.readout(z).squeeze(-1)

def _sample_truncated_gaussian(B: int, d: int, sigma: float, device, dtype, max_iters: int = 10) -> torch.Tensor:
    x = torch.randn(B, d, device=device, dtype=dtype) * sigma
    for _ in range(max_iters):
        mask = (x < -1.0) | (x > 1.0)
        if not mask.any():
            break
        x_new = torch.randn(B, d, device=device, dtype=dtype) * sigma
        x = torch.where(mask, x_new, x)
    return x.clamp(-1.0, 1.0)

def truncated_normal(shape, mean=0.0, std=0.2, low=-1.0, high=1.0, device=None, dtype=None):
    device = device or torch.device("cpu")
    dtype = dtype or torch.get_default_dtype()
    out = torch.empty(shape, device=device, dtype=dtype)
    flat = out.view(-1)
    mask = torch.ones(flat.numel(), dtype=torch.bool, device=device)
    while mask.any():
        k = mask.sum().item()
        z = torch.randn(k * 3, device=device, dtype=dtype) * std + mean
        z = z[(z >= low) & (z <= high)]
        if z.numel() == 0:
            continue
        take = min(z.numel(), k)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)[:take]
        flat[idx] = z[:take]
        mask[idx] = False
    return out

def sample_x(cfg: Config, B: int, device) -> torch.Tensor:
    if cfg.x_sampler == "uniform":
        return (torch.rand(B, cfg.d_in, device=device, dtype=cfg.dtype) * 2.0) - 1.0
    elif cfg.x_sampler == "normal":
        return torch.randn(B, cfg.d_in, device=device, dtype=cfg.dtype)
    elif cfg.x_sampler == "mixture":
        x_uni = (torch.rand(B, cfg.d_in, device=device, dtype=cfg.dtype) * 2.0) - 1.0
        x_trn = _sample_truncated_gaussian(
            B, cfg.d_in, cfg.trunc_normal_sigma, device, cfg.dtype, cfg.trunc_max_iters
        )
        choose_uni = (torch.rand(B, 1, device=device) < cfg.mixture_uniform_prob)
        return torch.where(choose_uni, x_uni, x_trn)
    elif cfg.x_sampler == "mixture2":
        N1 = B * 11 // 20
        N2 = B * 3 // 20
        N3 = B * 3 // 20
        N4 = B - N1 - N2 - N3
        s1 = (torch.rand(N1, cfg.d_in, device=device, dtype=cfg.dtype) * 2.0) - 1.0
        s2 = truncated_normal((N2, cfg.d_in), mean=0.0, std=0.2,low=-1.0, high=1.0, device=device, dtype=cfg.dtype)
        s3 = truncated_normal((N3, cfg.d_in), mean=-0.5, std=0.2, low=-1.0, high=1.0, device=device, dtype=cfg.dtype)
        s4 = truncated_normal((N4, cfg.d_in), mean=0.5, std=0.2, low=-1.0, high=1.0, device=device, dtype=cfg.dtype)
        return torch.cat([s1, s2, s3, s4], dim=0)
    else:
        raise ValueError("x_sampler must be 'uniform' or 'normal'")


@torch.no_grad()
def evaluate(model: nn.Module, cfg: Config, device) -> float:
    model.eval()
    # 生成一次样本
    x = sample_x(cfg, cfg.eval_points, device)
    y_true = true_function(x)
    y_pred = model(x)

    # 计算 L2 相对误差
    l2_error = torch.norm(y_pred - y_true, p=2) / torch.norm(y_true, p=2)

    model.train()
    return l2_error.item()


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, step: int, cfg: Config, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    chk = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "cfg": cfg.__dict__,
    }
    torch.save(chk, path)


def try_resume(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, device: torch.device, cfg: Config, ckpt_path: str | None = None) -> int:
    path = ckpt_path or cfg.ckpt_path
    if not (cfg.resume and os.path.exists(path)):
        return 1
    chk = torch.load(path, map_location=device)
    model.load_state_dict(chk["model_state"])  # type: ignore[arg-type]
    optimizer.load_state_dict(chk["optimizer_state"])  # type: ignore[arg-type]
    _optimizer_to_device(optimizer, device)
    if scheduler is not None and chk.get("scheduler_state") is not None:
        scheduler.load_state_dict(chk["scheduler_state"])  # type: ignore[arg-type]
    last_step = int(chk.get("step", 0))
    print(f"Resumed from {path} at step {last_step}.")
    return last_step + 1

@dataclass
class Config:
    # Data & training
    d_in: int = 2
    steps: int = 3000
    full_batch_size: int = 65536
    mini_batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 1e-3
    cosine_T_max: int = 50000
    print_every: int = 100
    eval_points: int = 1024
    x_sampler: str = "uniform"        # "uniform" or "normal"
    dtype: torch.dtype = torch.float32  # <<< change to torch.float32 if you prefer

    # Params
    m_base: int = 256
    n_scales: int = 4
    n_token: int = 32
    d_q: int = 32
    n_heads: int = 4
    n_queries: int = 1
    sigma: float = 0.1
    ffn_mult: int = 2
    fre_up: float = 300.0

    # ---- Checkpointing ----
    outdir: str = "checkpoints_cos"
    dot_ca_ckpt_path: str = "checkpoints_cos/dot_ca_last.pt"
    kernel_ca_ckpt_path: str = "checkpoints_cos/kernel_ca_last.pt"
    mlp_ckpt_path: str = "checkpoints_cos/mlp_last.pt"  # for mlp
    save_every: int = 1000
    resume: bool = True
cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, model_type, n_total_blocks, n_attn_blocks):
        set_seed(0)
        # Set default dtype BEFORE building the model
        torch.set_default_dtype(cfg.dtype)

        if model_type == "dot_ca":
            self.model = MultiscaleRFFCrossAttentionRegressor(
                d_in=cfg.d_in,
                m_base=cfg.m_base, n_scales=cfg.n_scales, n_token=cfg.n_token,
                d_q=cfg.d_q, n_heads=cfg.n_heads, n_queries=cfg.n_queries,
                n_total_blocks=n_total_blocks, n_attn_blocks=n_attn_blocks,
                sigma=cfg.sigma, ffn_mult=cfg.ffn_mult, dtype=cfg.dtype, device=device
            ).to(device)
            self.active_ckpt = cfg.dot_ca_ckpt_path
        elif model_type == "kernel_ca":
            self.model = KernelizedAttentionRegressor(
                d_in=cfg.d_in, n_token=cfg.n_token,d_q=cfg.d_q,
                n_total_blocks=n_total_blocks, n_attn_blocks=n_attn_blocks,
                ffn_mult=cfg.ffn_mult, fre_ub=cfg.fre_up, device=device
            ).to(device)
            self.active_ckpt = cfg.kernel_ca_ckpt_path

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.cosine_T_max)
        self.errors=[]
        self.full_x = sample_x(cfg, cfg.full_batch_size, device)
    def main(self):
        for step in range(cfg.steps):
            idx = torch.randperm(cfg.full_batch_size, device=device)[:cfg.mini_batch_size]
            x = self.full_x[idx, :]
            y = true_function(x)
            y_hat = self.model(x)
            loss = F.mse_loss(y_hat, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.sched.step()

            if step % cfg.print_every == 0:
                l2_relative_error = evaluate(self.model, cfg, device)
                self.errors.append(l2_relative_error)
                print(f"{step=},test_l2_relative_error {l2_relative_error:.4e}")


        l2_relative_error = evaluate(self.model, cfg, device)
        self.errors.append(l2_relative_error)
        print("\nFinal:")
        print(f"Test L2 Relative Error: {l2_relative_error:.4e}")

multiprocessing.set_start_method('spawn', force=True)
def train_task(model_type, i, result_queue):
    """
    单个训练任务：用Queue传递结果（spawn方式下Manager.list可能不稳定）
    """
    try:
        # 1. CUDA配置（移到Trainer初始化后，避免提前初始化）
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # 2. 核心训练逻辑（完全保留你的代码）
        print(f"\n=== Training {model_type} with {i} attention blocks ===")
        trainer = Trainer(model_type=model_type, n_total_blocks=4, n_attn_blocks=i)
        trainer.main()
        
        # 3. 将结果放入队列（代替shared_errors）
        result_queue.put((model_type, i, trainer.errors))
        
        # 4. 打印完成信息
        print(f" Finished training {model_type} - {i} blocks")
        
    except Exception as e:
        print(f" Error training {model_type} - {i} blocks: {str(e)}")
        result_queue.put((model_type, i, None))

if __name__ == "__main__":
    result_queue = multiprocessing.Queue()
    processes = []
    max_parallel = 8

    # 启动并行训练
    for i in range(1, 5):
        for model_type in ["dot_ca", "kernel_ca"]:
            # 控制并发数
            if len(processes) >= max_parallel:
                # 等待已有进程完成
                for p in processes:
                    p.join()
                processes = []
            
            # 启动新进程
            p = multiprocessing.Process(
                target=train_task,
                args=(model_type, i, result_queue)
            )
            p.start()
            processes.append(p)
            print(f" Started process: {model_type} - {i} blocks (PID: {p.pid})")
  
    # 等待所有进程完成
    for p in processes:
        p.join()

    errors = []
    while not result_queue.empty():
        res = result_queue.get()
        if res[2] is not None:  # 过滤失败的任务
            errors.append(res)


    step = list(range(0, cfg.steps + cfg.print_every, cfg.print_every))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    for model_type, i, error in errors:
        label = f"{i} attn"
        ax = axes[0] if model_type == 'dot_ca' else axes[1]
        ax.semilogy(step, error, label=label, alpha=0.7, linewidth=2)
    
    axes[0].set_title('dot_ca', fontsize=14)
    axes[0].set_xlabel("Training Step", fontsize=12)
    axes[0].set_ylabel("Test L2 Relative Error", fontsize=12)
    axes[0].legend()
    
    axes[1].set_title('kernel_ca', fontsize=14)
    axes[1].set_xlabel("Training Step", fontsize=12)
    axes[1].set_ylabel("Test L2 Relative Error", fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("l2_error_curve.png", dpi=300, bbox_inches='tight')

    with open("errors_list.pkl", "wb") as f:
        pickle.dump(errors, f)