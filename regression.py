"""
CRR_regressors_phase_cos.py
"""
from __future__ import annotations

import math, os, random
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# =========================
# Utilities & target
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def true_function(x: torch.Tensor) -> torch.Tensor:
    # Same toy target as in v8 (feel free to switch)
    #return torch.cos(100.234*(x*x).sum(-1))
    return torch.sign(torch.sqrt((x*x).sum(dim=-1))-0.5)


# =========================
# Multiscale RFF Tokenizer (scale-oriented) with cos+phase
# =========================
class MultiscaleRFFTokenizer(nn.Module):
    """
    Build multiscale RFF tokens. For base frequencies ω_m ∈ R^{d_in}, m=1..m_base,
    and scales k = 0..(n_scales-1), use ω_{m,k} = ω_m * 2^k  (matches v8).

    For each (m,k), create a single feature:  sqrt(2/M) * cos(ω_{m,k}^T x + b_{m,k}),
    where b_{m,k} ~ Uniform[0, 2π) are fixed random phases.

    We group 'group_size' such features into a token, then linear-project to d_model.
    Compared with the original [cos, sin] version (2 features per (m,k)), the
    projection input width per token halves from 2*group_size to group_size.
    """
    def __init__(
        self,
        d_in: int,
        m_base: int = 2048,
        n_scales: int = 4,
        group_size: int = 64,
        d_model: int = 128,
        sigma: float = 1.0,
        learnable_axis_scales: bool = True,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert n_scales >= 1
        self.d_in = d_in
        self.m_base = m_base
        self.n_scales = n_scales
        self.group_size = group_size
        self.d_model = d_model

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base frequencies ω_m ~ N(0, I / sigma^2)  (fixed buffer)
        self.register_buffer(
            "omega_base",
            torch.randn(m_base, d_in, dtype=dtype, device=device) / float(sigma),
        )

        # Optional axis-wise learnable scaling
        self.log_axis_scale = (
            nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))
            if learnable_axis_scales else None
        )

        # Total (m,k) count (before padding)
        self.m_total = m_base * n_scales

        # Pad so m_total is divisible by group_size
        if self.m_total % group_size != 0:
            pad = group_size - (self.m_total % group_size)
            self.register_buffer(
                "omega_base_pad",
                torch.randn(pad, d_in, dtype=dtype, device=device) / float(sigma),
            )
            self.m_total_padded = self.m_total + pad
            self.pad = pad
        else:
            self.register_buffer("omega_base_pad", torch.empty(0, d_in, dtype=dtype, device=device))
            self.m_total_padded = self.m_total
            self.pad = 0

        # ---- NEW: random phases for each (m,k) row in the concatenated bank ----
        # Build per-scale phase segments now so we don't regenerate every forward pass.
        phase_segments = []
        base_len = m_base + (self.pad if self.pad > 0 else 0)
        for _ in range(n_scales):
            # uniform phases in [0, 2π)
            phase_segments.append(2.0 * math.pi * torch.rand(base_len, dtype=dtype, device=device))
        phase_full = torch.cat(phase_segments, dim=0)  # length = n_scales * base_len
        self.register_buffer("phase_full", phase_full)  # will be sliced later if padded
        # ------------------------------------------------------------------------

        # Project grouped features to d_model (note: input width is group_size now)
        #self.proj = nn.Linear(group_size, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=True)

        # Stabilize feature magnitude.
        # With cos+random phase, Var[cos(• + b)] = 1/2 over b, so use sqrt(2/M).
        self.scale = math.sqrt(2.0 / float(self.m_total))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_in)
        returns: tokens (B, n_tokens, d_model)
        """
        B = x.shape[0]
        axis_scale = self.log_axis_scale.exp() if self.log_axis_scale is not None else None

        base = self.omega_base
        if self.pad > 0:
            base = torch.cat([base, self.omega_base_pad], dim=0)  # (m_base+pad, d_in)

        # Build scaled frequency bank across dyadic-like scales (v8 uses 2**k)
        scaled_omegas = []
        for k in range(self.n_scales):
            wk = base * (2**k)  # matches CRR_regressor_v8.py
            if axis_scale is not None:
                wk = wk * axis_scale
            scaled_omegas.append(wk)
        Omega = torch.cat(scaled_omegas, dim=0)  # (m_total_padded, d_in)

        # Project x onto the frequency bank and add fixed random phases
        # z_raw: (B, m_total_padded)
        z_raw = x @ Omega.t()

        # Slice phases to match unpadded m_total (so we can safely drop pad later)
        phases = self.phase_full  # length = n_scales*(m_base+pad)
        if self.pad > 0:
            z = z_raw[:, :self.m_total] + phases[:self.m_total]
        else:
            z = z_raw + phases  # (B, m_total)

        # Single feature per (m,k): scaled cosine with random phase
        feats = torch.cos(z) * self.scale  # (B, m_total)

        # Group per token
        n_tokens = self.m_total // self.group_size
        tokens_in = feats.view(B, n_tokens, self.group_size)  # (B, n_tokens, group_size)

        # Project + normalize
        H = tokens_in  # (B, n_tokens, group_size)
        #H = self.proj(tokens_in)     # (B, n_tokens, d_model)
        H = self.norm(H)
        return H


# =========================
# Cross-Attention Residual Block (unchanged)
# =========================
class CrossAttentionResBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, ffn_mult: int = 2):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        #self.ln_q = nn.LayerNorm(d_model)
        #self.ln_kv = nn.LayerNorm(d_model)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        #self.ln_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, Q: torch.Tensor, KV: torch.Tensor) -> torch.Tensor:
        B, L, _ = Q.shape
        #kv = self.ln_kv(KV)
        #q = self.ln_q(Q)

        kv = KV
        q = Q

        q = self.W_q(q).view(B, L, self.n_heads, self.d_head).transpose(1, 2)              # (B, H, L, Dh)
        k = self.W_k(kv).view(B, KV.shape[1], self.n_heads, self.d_head).transpose(1, 2)   # (B, H, T, Dh)
        v = self.W_v(kv).view(B, KV.shape[1], self.n_heads, self.d_head).transpose(1, 2)   # (B, H, T, Dh)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)        # (B, H, L, T)
        attn = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn, v)                                                    # (B, H, L, Dh)
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)            # (B, L, D)

        Q = Q + self.W_o(context)
        #Q = Q + self.ffn(self.ln_ff(Q))
        Q = Q + self.ffn(Q)
        return Q


# =========================
# Models (unchanged)
# =========================
class MultiscaleRFFCrossAttentionRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        m_base: int = 2048,
        n_scales: int = 4,
        group_size: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_queries: int = 8,
        n_blocks: int = 3,
        sigma: float = 1.0,
        learnable_axis_scales: bool = True,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = MultiscaleRFFTokenizer(
            d_in=d_in, m_base=m_base, n_scales=n_scales, group_size=group_size,
            d_model=d_model, sigma=sigma, learnable_axis_scales=learnable_axis_scales,
            dtype=dtype, device=device
        )
        self.d_model = d_model
        self.n_queries = n_queries
        #self.query_tokens = nn.Parameter(torch.randn(n_queries, d_model, dtype=dtype, device=device) * 0.02)

        # L1(x): input-conditioned query features (B, d_model)
        self.L1 = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.GELU(),
        )

        # Expand to queries (B, n_queries, d_model)
        if n_queries > 1:
            self.q_from_l1 = nn.Linear(d_model, n_queries * d_model)
        else:
            self.q_from_l1 = None

        self.blocks = nn.ModuleList([CrossAttentionResBlock(d_model, n_heads, 4) for _ in range(n_blocks)])

        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = self.tokenizer(x)                     # (B, n_tokens, d_model)

        l1 = self.L1(x) 
        if self.n_queries == 1:
            Q = l1.unsqueeze(1)                         # (B, 1, d_model)
        else:
            Q = self.q_from_l1(l1).view(x.size(0), self.n_queries, self.d_model)

        #B = x.shape[0]
        #Q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            Q = blk(Q, H)

        z = Q.mean(dim=1)
        return self.readout(z).squeeze(-1)


class MLPRegressor(nn.Module):
    """Simple MLP baseline."""
    def __init__(self, d_in: int, widths: Tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        layers = []
        in_dim = d_in
        for w in widths:
            layers += [nn.Linear(in_dim, w), nn.Tanh()]
            in_dim = w
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# =========================
# Training / Eval (unchanged)
# =========================
@dataclass
class Config:
    # Data & training
    d_in: int = 2
    steps: int = 50000
    batch_size: int = 5000
    lr: float = 1e-3
    weight_decay: float = 0.0
    cosine_T_max: int = 50000
    print_every: int = 200
    test_batches: int = 20
    x_sampler: str = "uniform"        # "uniform" or "normal"
    dtype: torch.dtype = torch.float64  # <<< change to torch.float32 if you prefer

    # Model selector
    model_type: str = "ms_rff_cross" # "ms_rff_cross" | "mlp"
    mlp_widths: Tuple[int, ...] = (256, 256, 256)

    # Multiscale-RFF CrossAttention params
    m_base: int = 64
    n_scales: int = 3
    group_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_queries: int = 1
    n_blocks: int = 4
    sigma: float = 1
    learnable_axis_scales: bool = True

    # ---- Checkpointing ----
    outdir: str = "checkpoints_cos"
    ckpt_path: str = "checkpoints_cos/last.pt"       # for rff_cross
    mlp_ckpt_path: str = "checkpoints_cos/mlp_last.pt"  # for mlp
    save_every: int = 1000
    resume: bool = True


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
def evaluate(model: nn.Module, cfg: Config, device) -> Tuple[float, float]:
    model.eval()
    mse_list, mae_list = [], []
    for _ in range(cfg.test_batches):
        x = sample_x(cfg, cfg.batch_size, device)
        y = true_function(x)
        y_hat = model(x)
        mse_list.append(F.mse_loss(y_hat, y).item())
        mae_list.append(F.l1_loss(y_hat, y).item())
    model.train()
    return sum(mse_list)/len(mse_list), sum(mae_list)/len(mae_list)


def plot_results(model: nn.Module, cfg: Config, device, tag: str):
    model.eval()
    rs = torch.linspace(-1.0, 1.0, 2001, device=device, dtype=cfg.dtype)
    #true_vals = torch.cos(100.234*rs*rs*cfg.d_in)
    true_vals = torch.sign(torch.sqrt((rs*rs*cfg.d_in))-0.5)
    xs = torch.stack([rs for _ in range(cfg.d_in)], dim=1)
    with torch.no_grad():
        approx_vals = model(xs).cpu()
    plt.figure(figsize=(6,4))
    plt.plot(rs.cpu(), true_vals.cpu(), label="True f(r)")
    plt.plot(rs.cpu(), approx_vals, label="Approx f(r)", linestyle="--")
    plt.xlabel("r")
    plt.ylabel("f(r)")
    plt.legend()
    plt.title("Comparison of true vs approximation")
    plt.tight_layout()
    fig_path = os.path.join(cfg.outdir, f"prediction_{tag}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()


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


def main():
    set_seed(0)
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set default dtype BEFORE building the model
    torch.set_default_dtype(cfg.dtype)

    # Build model AFTER setting default dtype so all layers/params use cfg.dtype
    if cfg.model_type == "mlp":
        model = MLPRegressor(d_in=cfg.d_in, widths=cfg.mlp_widths).to(device)
        active_ckpt = cfg.mlp_ckpt_path
    else:
        model = MultiscaleRFFCrossAttentionRegressor(
            d_in=cfg.d_in,
            m_base=cfg.m_base, n_scales=cfg.n_scales, group_size=cfg.group_size,
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_queries=cfg.n_queries, n_blocks=cfg.n_blocks,
            sigma=cfg.sigma, learnable_axis_scales=cfg.learnable_axis_scales,
            dtype=cfg.dtype, device=device
        ).to(device)
        active_ckpt = cfg.ckpt_path

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.cosine_T_max)

    start_step = try_resume(model, opt, sched, device, cfg, ckpt_path=active_ckpt)
    if start_step > cfg.steps:
        print(f"Checkpoint step {start_step-1} >= target steps {cfg.steps}. Skipping training.")
    else:
        for step in range(start_step, cfg.steps + 1):
            x = sample_x(cfg, cfg.batch_size, device)
            y = true_function(x)
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % cfg.print_every == 0 or step == 1:
                mse, mae = evaluate(model, cfg, device)
                print(f"step {step:5d} | train_mse {loss.item():.4e} | test_mse {mse:.4e} | test_mae {mae:.4e}")

            if step % cfg.save_every == 0:
                save_checkpoint(model, opt, sched, step, cfg, active_ckpt)
                print(f"Saved checkpoint to {active_ckpt} at step {step}.")
                plot_results(model, cfg, device, tag=f"step{step}")

    # Always save final
    save_checkpoint(model, opt, sched, cfg.steps, cfg, active_ckpt)
    print(f"Saved final checkpoint to {active_ckpt}.")

    # Eval and plot
    mse, mae = evaluate(model, cfg, device)
    print("\nFinal:")
    print(f"Test MSE: {mse:.4e} | Test MAE: {mae:.4e}")
    #plot_results(model, cfg, device)


if __name__ == "__main__":
    main()
