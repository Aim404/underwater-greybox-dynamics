#!/usr/bin/env python3
"""
v21 pure-dynamics model — grey box (Port-Hamiltonian Ocean Network with Inertial Context)

架构:
  Input: 历史窗口 (h步) 的 [状态, 控制, 环境感知]
    ↓
  PhysicsParamNet: 预测水动力参数 (附加质量, 阻尼, 恢复力, 推力器映射)
    ↓
  Fossen ODE: 用预测参数通过物理方程计算状态导数
    ↓                      ↑
  ResidualGRU: 捕捉未建模动力学 (延迟, 涡流, 碰撞瞬态)
    ↓
  RK4积分 → 下一状态

参考:
  - FHNN (Zhang et al., 2025): 参数-方程耦合架构
  - Hamiltonian Neural ODE (Jin et al., 2025): 能量结构
  - NeRD (Xu et al., 2025): 历史窗口 + 碰撞混合框架
  - DeepVL (Singh & Alexis, 2025): 本体感知 + GRU
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# 基础组件
# ═══════════════════════════════════════════════════════════════════

class CholeskyHead(nn.Module):
    """
    输出 Cholesky 下三角 L，保证 L @ L^T 正定对称
    用于附加质量矩阵和阻尼矩阵的参数化
    """
    def __init__(self, in_dim, out_dim=21):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, out_dim),
        )
        # 下三角索引映射 (6x6)
        self._tril_indices = []
        idx = 0
        for i in range(6):
            for j in range(i + 1):
                self._tril_indices.append((i, j, idx))
                idx += 1

    def forward(self, x):
        raw = self.mlp(x)  # (batch, 21)
        B = raw.shape[0]
        L = torch.zeros(B, 6, 6, device=raw.device, dtype=raw.dtype)
        for i, j, idx in self._tril_indices:
            if i == j:
                L[:, i, j] = F.softplus(raw[:, idx]) + 1e-4  # 对角线 > 0
            else:
                L[:, i, j] = raw[:, idx]
        return L @ L.transpose(-1, -2)  # (batch, 6, 6) 正定对称


class SymmetricHead(nn.Module):
    """输出对称半正定矩阵 (用于二次阻尼)"""
    def __init__(self, in_dim, out_dim=21):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, out_dim),
        )
        self._tril_indices = []
        idx = 0
        for i in range(6):
            for j in range(i + 1):
                self._tril_indices.append((i, j, idx))
                idx += 1

    def forward(self, x):
        raw = self.mlp(x)  # (batch, 21)
        B = raw.shape[0]
        L = torch.zeros(B, 6, 6, device=raw.device, dtype=raw.dtype)
        for i, j, idx in self._tril_indices:
            if i == j:
                L[:, i, j] = F.softplus(raw[:, idx])  # 非负对角线
            else:
                L[:, i, j] = raw[:, idx]
        return L @ L.transpose(-1, -2)  # 半正定


class FiLMConditioner(nn.Module):
    """FiLM 条件化: γ * features + β"""
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.gamma_net = nn.Linear(cond_dim, feature_dim)
        self.beta_net = nn.Linear(cond_dim, feature_dim)

    def forward(self, features, cond):
        gamma = self.gamma_net(cond)
        beta = self.beta_net(cond)
        return gamma * features + beta


# ═══════════════════════════════════════════════════════════════════
# Physics Param Network
# ═══════════════════════════════════════════════════════════════════

def _inv_softplus(y, eps=1e-4):
    """x 使得 softplus(x) + eps ≈ y"""
    y = max(float(y) - eps, 1e-6)
    if y >= 10.0:
        return y  # softplus(x) ≈ x for large x
    return math.log(math.exp(y) - 1.0)


def _init_heads_to_gazebo(phys_net):
    """
    用 Gazebo 真值初始化 PhysicsParamNet 的各个预测头

    通过设置 head bias, 使网络在初始化时输出接近 Gazebo 参数,
    加速收敛并防止训练早期因参数偏差导致的动力学失真.

    关键: DL_vec/DQ_vec 的 bias 设的是 damping_scale 缩放前的值,
    forward_step 中会乘以 damping_scale 得到最终值.
    """
    # Gazebo 线性阻尼 (绝对值), 需除以 damping_scale 得到 raw 值
    # damping_scale_linear=10, damping_scale_angular=30 (来自 model_v2 默认)
    _gazebo_dl = [44.2, 41.0, 429.9, 55.3, 63.7, 21.6]
    _gazebo_dq = [260.7, 345.9, 634.6, 81.6, 94.1, 63.6]
    _gazebo_ma_diag = [160.4, 251.3, 752.6, 38.3, 60.4, 16.1]
    _dl_scale = [10.0, 10.0, 10.0, 30.0, 30.0, 30.0]
    _dq_scale = [10.0, 10.0, 10.0, 30.0, 30.0, 30.0]

    with torch.no_grad():
        # ── head_DL bias: 使 DL_vec_raw ≈ Gazebo / damping_scale ──
        dl_bias = torch.zeros(6)
        for i in range(6):
            raw_target = _gazebo_dl[i] / _dl_scale[i]
            dl_bias[i] = _inv_softplus(raw_target)
        phys_net.head_DL.bias.copy_(dl_bias)

        # ── head_DQ bias: 同理 ──
        dq_bias = torch.zeros(6)
        for i in range(6):
            raw_target = _gazebo_dq[i] / _dq_scale[i]
            dq_bias[i] = _inv_softplus(raw_target)
        phys_net.head_DQ.bias.copy_(dq_bias)

        # ── CholeskyHead bias: 初始化为 Gazebo 附加质量对角线 ──
        # L @ L^T = MA_diag → L_ii = sqrt(MA_diag[i]), L_ij = 0 (i≠j)
        # CholeskyHead 输出 21 维 raw, 然后映射为下三角 L
        _tril_indices = []
        idx = 0
        for i in range(6):
            for j in range(i + 1):
                _tril_indices.append((i, j, idx))
                idx += 1

        ma_bias = torch.zeros(21)
        for i, j, idx in _tril_indices:
            if i == j:
                target_l = math.sqrt(max(_gazebo_ma_diag[i], 1e-4))
                ma_bias[idx] = _inv_softplus(target_l)
            else:
                ma_bias[idx] = 0.0  # 非对角初始为零
        # CholeskyHead 的最后一层是 Linear(in_dim, 21)
        last_linear = phys_net.head_MA.mlp[-1]
        if isinstance(last_linear, nn.Linear) and last_linear.out_features == 21:
            last_linear.bias.data.copy_(ma_bias)


# ═══════════════════════════════════════════════════════════════════
# Physics Param Network
# ═══════════════════════════════════════════════════════════════════

class PhysicsParamNet(nn.Module):
    """
    从历史窗口预测水动力参数

    输入: 历史特征 (batch, h, input_dim) + 负载条件
    输出: Fossen方程中的关键参数
    """
    def __init__(self, input_dim=36, hidden_dim=256, hist_len=10,
                 cond_dim=8):
        super().__init__()
        # 时序特征提取 (1D卷积)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 负载条件嵌入
        self.cond_embed = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
        )

        # FiLM 条件化
        self.film = FiLMConditioner(cond_dim, hidden_dim)

        # 共享 MLP
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # 分头预测
        self.head_MA = CholeskyHead(hidden_dim, 21)    # 附加质量 (6x6 正定)
        self.head_DL = nn.Linear(hidden_dim, 6)         # 线性阻尼 (6维对角)
        self.head_DQ = nn.Linear(hidden_dim, 6)         # 二次阻尼 (6维对角)
        self.head_gravity = nn.Linear(hidden_dim, 8)    # 重力/浮力参数
        self.head_thruster = nn.Linear(hidden_dim, 48)  # 推力器映射 (6x8)

        # ── Gazebo 真值初始化 ──
        _init_heads_to_gazebo(self)

        # TAM 先验: 6x8 推力分配矩阵展平为 bias (来自 TAM.yaml)
        _tam = torch.tensor([
            [-0.707107,  0.707106, -0.707107,  0.707107,  0.0,       0.0,       0.0,       0.0      ],
            [-0.707106, -0.707107,  0.707107,  0.707107,  0.258819,  0.258819, -0.258819, -0.258819],
            [ 0.0,       0.0,       0.0,       0.0,       0.965926,  0.965926,  0.965926,  0.965926 ],
            [ 0.005657,  0.005657, -0.005657, -0.005657, -0.443678, -0.443678,  0.443678,  0.443678],
            [-0.005657,  0.005657, -0.005657,  0.005657, -0.507111,  0.507111, -0.507111,  0.507111],
            [-0.555079,  0.555079,  0.555079, -0.555079,  0.135880, -0.135880, -0.135880,  0.135880],
        ], dtype=torch.float32).flatten()
        nn.init.normal_(self.head_thruster.weight, std=0.01)
        nn.init.constant_(self.head_thruster.bias, 0.0)
        self.head_thruster.bias.data.copy_(_tam * 1.0)  # 完整 TAM 先验

    def forward(self, x_hist, payload=None):
        """
        Args:
            x_hist: (batch, h, input_dim) 历史窗口
            payload: (batch, 1) 负载标识

        Returns:
            dict of physics parameters
        """
        # 时序特征
        x_t = x_hist.transpose(1, 2)               # (batch, input_dim, h)
        z = self.temporal_conv(x_t).squeeze(-1)     # (batch, hidden_dim)

        # FiLM 条件化
        if payload is not None:
            cond = self.cond_embed(payload)          # (batch, cond_dim)
            z = self.film(z, cond)

        z = self.shared(z)

        MA = self.head_MA(z)          # (batch, 6, 6)
        DL_vec = F.softplus(self.head_DL(z)) + 1e-4  # (batch, 6) 正对角线
        DQ_vec = F.softplus(self.head_DQ(z))          # (batch, 6) 非负对角线
        DL = torch.diag_embed(DL_vec)                 # (batch, 6, 6)
        DQ = torch.diag_embed(DQ_vec)                 # (batch, 6, 6)
        grav = self.head_gravity(z)   # (batch, 8)
        T = self.head_thruster(z)     # (batch, 48)
        T = T.view(-1, 6, 8)         # (batch, 6, 8)

        return {
            'MA': MA, 'DL': DL, 'DQ': DQ, 'DL_vec': DL_vec, 'DQ_vec': DQ_vec,
            'gravity': grav, 'thruster_matrix': T,
        }


# ═══════════════════════════════════════════════════════════════════
# Fossen ODE Integrator
# ═══════════════════════════════════════════════════════════════════

def quat_to_rotation_matrix(q):
    """四元数 → 旋转矩阵 (x,y,z,w 顺序)"""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def _skew(v):
    """3D 向量的反对称矩阵 (batch)"""
    B = v.shape[0]
    S = torch.zeros(B, 3, 3, device=v.device, dtype=v.dtype)
    S[:, 0, 1] = -v[:, 2]
    S[:, 0, 2] = v[:, 1]
    S[:, 1, 0] = v[:, 2]
    S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]
    S[:, 2, 1] = v[:, 0]
    return S


def construct_coriolis(M, nu):
    """
    从全惯性矩阵 M 和速度 nu 构造科里奥利-向心力矩阵 (Fossen 标准公式)

    C(v) = [   0₃ₓ₃    ,  -S(M₁₁ v₁ + M₁₂ v₂)  ]
           [ -S(M₁₁ v₁ + M₁₂ v₂) , -S(M₂₁ v₁ + M₂₂ v₂) ]

    其中 S(·) 是 3x3 反对称矩阵
    """
    B = M.shape[0]
    v1 = nu[:, :3]
    v2 = nu[:, 3:6]

    M11 = M[:, :3, :3]
    M12 = M[:, :3, 3:6]
    M21 = M[:, 3:6, :3]
    M22 = M[:, 3:6, 3:6]

    Mv = torch.bmm(M, nu.unsqueeze(-1)).squeeze(-1)
    a1 = Mv[:, :3]
    a2 = Mv[:, 3:6]

    C = torch.zeros(B, 6, 6, device=M.device, dtype=M.dtype)
    C[:, :3, 3:6] = -_skew(a1)
    C[:, 3:6, :3] = -_skew(a1)
    C[:, 3:6, 3:6] = -_skew(a2)

    return C


def compute_restore_force(quat, grav_params):
    """
    计算恢复力 (重力+浮力)，使用旋转矩阵避免欧拉角奇异性

    g(η) = -[ f_G^b + f_B^b ; r_G × f_G^b + r_B × f_B^b ]

    其中 f_G^b = R^T * [0, 0, W], f_B^b = R^T * [0, 0, -B]

    grav_params: (batch, 8) = [W, B, rg_x, rg_y, rg_z, rb_x, rb_y, rb_z]
    """
    B = quat.shape[0]
    W = grav_params[:, 0].abs() + 10.0
    Buoy = grav_params[:, 1].abs() + 10.0
    rg = grav_params[:, 2:5]
    rb = grav_params[:, 5:8]

    R = quat_to_rotation_matrix(quat)

    f_G_world = torch.zeros(B, 3, device=quat.device, dtype=quat.dtype)
    f_G_world[:, 2] = W
    f_G_body = torch.bmm(R.transpose(-1, -2), f_G_world.unsqueeze(-1)).squeeze(-1)

    f_B_world = torch.zeros(B, 3, device=quat.device, dtype=quat.dtype)
    f_B_world[:, 2] = -Buoy
    f_B_body = torch.bmm(R.transpose(-1, -2), f_B_world.unsqueeze(-1)).squeeze(-1)

    g_force = torch.zeros(B, 6, device=quat.device, dtype=quat.dtype)
    g_force[:, :3] = -(f_G_body + f_B_body)
    g_force[:, 3:6] = -(torch.cross(rg, f_G_body, dim=-1) + torch.cross(rb, f_B_body, dim=-1))

    return g_force


def fossen_dynamics_step(nu, quat, params, thruster_input, residual=None):
    """
    Fossen 6-DOF 方程的一步计算

    Args:
        nu: (batch, 6) 机体系速度 [u,v,w,p,q,r]
        quat: (batch, 4) 四元数 [x,y,z,w]
        params: dict from PhysicsParamNet
        thruster_input: (batch, 8) 推力器指令
        residual: (batch, 6) 残差加速度

    Returns:
        nu_dot: (batch, 6) 机体系加速度
        eta_dot: (batch, 7) 世界系位姿变化 [dx,dy,dz, dqx,dqy,dqz,dqw]
    """
    MA = params['MA']              # (batch, 6, 6)
    DL_vec = params['DL_vec']     # (batch, 6) 线性阻尼对角线
    DQ_vec = params['DQ_vec']     # (batch, 6) 二次阻尼对角线
    T = params['thruster_matrix'] # (batch, 6, 8)

    # 惯性矩阵 + 质量归一化 (避免 M 条件数过大导致梯度压缩)
    # DLROV 真实总质量: 刚性500kg + 附加质量 [160,251,753] = [660,751,1253]
    # 真实总惯量: 刚性 [140,260,220] + 附加 [38,60,16] = [179,320,236]
    m_scale = torch.tensor([660.0, 751.0, 1253.0, 179.0, 320.0, 236.0],
                            device=nu.device, dtype=nu.dtype)
    M_RB = torch.diag_embed(m_scale).unsqueeze(0).expand(nu.shape[0], -1, -1)
    M_raw = M_RB + MA

    # 质量归一化: M_scaled = diag(1/m_scale) @ M_raw, rhs_scaled = rhs / m_scale
    # 等价变换: nu_dot = M_raw^{-1} @ rhs = M_scaled^{-1} @ rhs_scaled
    # 此时 M_scaled ≈ I, 梯度不被压缩
    inv_scale = 1.0 / m_scale
    M = M_raw * inv_scale.unsqueeze(-1)

    # 科里奥利矩阵 (用原始 M_raw 构造，再缩放)
    C_raw = construct_coriolis(M_raw, nu)
    C = C_raw * inv_scale.unsqueeze(-1)

    # 阻尼力: D(v) = D_L + D_Q @ diag(|v|) → 元素级乘法（对角阵）
    damping_raw = DL_vec * nu + DQ_vec * nu.abs() * nu
    damping = damping_raw * inv_scale

    # 恢复力 (缩放)
    g_force_raw = compute_restore_force(quat, params['gravity'])
    g_force = g_force_raw * inv_scale

    # 推力器力/力矩 (缩放)
    tau_raw = torch.bmm(T, thruster_input.unsqueeze(-1)).squeeze(-1)
    tau = tau_raw * inv_scale

    # Fossen方程 (归一化): M_scaled * nu_dot_phys = rhs_scaled
    rhs = tau - torch.bmm(C, nu.unsqueeze(-1)).squeeze(-1) - damping - g_force
    rhs = torch.clamp(rhs, -50.0, 50.0)

    nu_dot_phys = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
    nu_dot_phys = torch.clamp(nu_dot_phys, -5.0, 5.0)

    # 替换 NaN 为 0 (应对 ill-conditioned M)
    nu_dot_phys = torch.nan_to_num(nu_dot_phys, nan=0.0, posinf=5.0, neginf=-5.0)

    # 残差直接加在加速度空间 (避免梯度被 M^{-1} 压缩)
    if residual is not None:
        residual = torch.clamp(residual, -3.0, 3.0)
        nu_dot = nu_dot_phys + residual
    else:
        nu_dot = nu_dot_phys

    # 运动学方程: eta_dot = J(eta) * nu
    R = quat_to_rotation_matrix(quat)  # (batch, 3, 3)
    eta_dot_pos = torch.bmm(R, nu[:, :3].unsqueeze(-1)).squeeze(-1)  # (batch, 3)

    # 四元数导数 (简化)
    omega = nu[:, 3:6]
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    dq = torch.stack([
        0.5 * (qw * omega[:, 0] + qz * omega[:, 1] - qy * omega[:, 2]),
        0.5 * (-qz * omega[:, 0] + qw * omega[:, 1] + qx * omega[:, 2]),
        0.5 * (qy * omega[:, 0] - qx * omega[:, 1] + qw * omega[:, 2]),
        0.5 * (-qx * omega[:, 0] - qy * omega[:, 1] - qz * omega[:, 2]),
    ], dim=-1)

    eta_dot = torch.cat([eta_dot_pos, dq], dim=-1)  # (batch, 7)

    return nu_dot, eta_dot


# ═══════════════════════════════════════════════════════════════════
# Residual GRU Network
# ═══════════════════════════════════════════════════════════════════

class ResidualGRUNet(nn.Module):
    """
    捕捉 Fossen 方程未能建模的动力学效应

    输入: 历史窗口 (batch, h, input_dim)
    输出: 残差加速度 (batch, 6)
    """
    def __init__(self, input_dim=36, hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 6),
        )
        # 初始化输出接近0 (残差初始小)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x_hist):
        """
        Args:
            x_hist: (batch, h, input_dim)
        Returns:
            residual: (batch, 6)
        """
        _, h_n = self.gru(x_hist)
        return self.mlp(h_n[-1])


# ═══════════════════════════════════════════════════════════════════
# 完整模型
# ═══════════════════════════════════════════════════════════════════

class GreyBoxDynamics(nn.Module):
    """
    Grey-box dynamics: Port-Hamiltonian structure with Inertial Context

    输入: 历史窗口的特征 (状态+控制+环境)
    输出: 下一时刻的状态预测

    状态定义:
        position:  (3,) 世界系位置
        orientation: (4,) 四元数 [x,y,z,w]
        lin_vel:  (3,) 机体系线速度
        ang_vel:  (3,) 机体系角速度
    """
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}

        # 状态维度
        self.pos_dim = 3
        self.quat_dim = 4
        self.vel_dim = 6   # 3 lin + 3 ang
        self.state_dim = self.pos_dim + self.quat_dim + self.vel_dim  # 13
        self.thruster_dim = cfg.get('thruster_dim', 8)
        self.thruster_ode_dim = 8  # ODE 始终用 8 通道 (T: 6×8)
        self.env_dim = cfg.get('env_dim', 9)  # depth(1) + altitude(1) + wall_dist(6) + contact_mag(1)
        self.payload_dim = 1
        self.hist_len = cfg.get('hist_len', 10)
        self.jump_frame = cfg.get('jump_frame', 0)
        self.target_hz = cfg.get('target_hz', 15.0)

        # 总输入维度 (用于 PhysicsParamNet 和 ResidualGRU)
        self.input_dim = self.state_dim + self.thruster_dim + self.env_dim + self.payload_dim  # 31

        # 子模块
        self.physics_net = PhysicsParamNet(
            input_dim=self.input_dim,
            hidden_dim=cfg.get('physics_hidden', 256),
            hist_len=self.hist_len,
            cond_dim=cfg.get('cond_dim', 8),
        )

        self.residual_gru = ResidualGRUNet(
            input_dim=self.input_dim,
            hidden_dim=cfg.get('residual_hidden', 128),
            num_layers=cfg.get('residual_layers', 2),
        )

        # 归一化参数 (外部设置)
        self.register_buffer('_norm_mean', torch.zeros(self.input_dim), persistent=False)
        self.register_buffer('_norm_std', torch.ones(self.input_dim), persistent=False)

    def set_norm_stats(self, mean, std):
        self._norm_mean.copy_(mean)
        self._norm_std.copy_(std)

    def _normalize(self, x):
        return (x - self._norm_mean) / self._norm_std.clamp(min=1e-4)

    def _build_input(self, position, orientation, lin_vel, ang_vel,
                     thruster, depth, altitude, wall_dist, contact_mag, payload):
        raw = torch.cat([
            position, orientation, lin_vel, ang_vel,
            thruster, depth, altitude, wall_dist, contact_mag, payload
        ], dim=-1)
        return self._normalize(raw)

    def _build_hist_input(self, batch, t_start, t_end):
        """构建历史窗口输入 (batch, h, input_dim)"""
        B = batch['position'].shape[0]
        h = t_end - t_start

        pos = batch['position'][:, t_start:t_end]
        quat = batch['orientation'][:, t_start:t_end]
        lv = batch['lin_vel'][:, t_start:t_end]
        av = batch['ang_vel'][:, t_start:t_end]
        thr = batch['thruster'][:, t_start:t_end]
        dep = batch['depth'][:, t_start:t_end]
        alt = batch['altitude'][:, t_start:t_end]
        wd = batch['wall_dist'][:, t_start:t_end]
        cm = batch['contact_mag'][:, t_start:t_end]

        # payload: 标量标签，无时间维度 → 广播到 (B, h, 1)
        pay_raw = batch['payload']
        if pay_raw.dim() == 2:
            pay = pay_raw.unsqueeze(1).expand(B, h, 1)
        elif pay_raw.dim() == 3:
            pay = pay_raw[:, t_start:t_end]
        elif pay_raw.dim() == 1:
            pay = pay_raw.unsqueeze(-1).unsqueeze(1).expand(B, h, 1)
        else:
            pay = pay_raw.view(B, 1).unsqueeze(1).expand(B, h, 1)

        raw = torch.cat([pos, quat, lv, av, thr, dep, alt, wd, cm, pay], dim=-1)
        return self._normalize(raw)

    def _get_current_state(self, batch, t):
        """获取 t 时刻的状态"""
        pos = batch['position'][:, t]
        quat = batch['orientation'][:, t]
        lv = batch['lin_vel'][:, t]
        av = batch['ang_vel'][:, t]
        return pos, quat, torch.cat([lv, av], dim=-1)

    def forward_step(self, position, quat, nu, thruster_input,
                     depth, altitude, wall_dist, contact_mag, payload,
                     hist_input):
        """
        单步前向

        Args:
            position: (batch, 3)
            quat: (batch, 4)
            nu: (batch, 6) 机体系速度
            thruster_input: (batch, 8)
            depth, altitude, wall_dist, contact_mag: (batch, env_subdim)
            payload: (batch, 1)
            hist_input: (batch, h, input_dim) 历史窗口

        Returns:
            next_position: (batch, 3)
            next_quat: (batch, 4) 归一化后
            next_nu: (batch, 6)
            nu_dot: (batch, 6)
            params: dict
        """
        # 1. 预测物理参数
        params = self.physics_net(hist_input, payload)

        # 2. 预测残差
        residual = self.residual_gru(hist_input)

        # 推力器: 若 jump_frame>0, hist_input 含展开推力 (B, dt_mult*8), ODE 用均值
        if self.thruster_dim > 8:
            dt_mult = self.jump_frame + 1
            thr_ode = thruster_input.view(-1, dt_mult, 8).mean(dim=1)
        else:
            thr_ode = thruster_input

        # 3. Fossen方程计算加速度
        nu_dot, eta_dot = fossen_dynamics_step(
            nu, quat, params, thr_ode, residual)

        # 4. RK4 积分
        dt = (self.jump_frame + 1) / self.target_hz

        def _ode_rhs(nu_s, quat_s, pos_s):
            nu_dot_s, eta_dot_s = fossen_dynamics_step(
                nu_s, quat_s, params, thr_ode, residual)
            return nu_dot_s, eta_dot_s

        # k1
        k1_nu, k1_eta = _ode_rhs(nu, quat, position)
        # k2
        nu2 = nu + 0.5 * dt * k1_nu
        quat2 = F.normalize(quat + 0.5 * dt * k1_eta[:, 3:7], dim=-1)
        pos2 = position + 0.5 * dt * k1_eta[:, :3]
        k2_nu, k2_eta = _ode_rhs(nu2, quat2, pos2)
        # k3
        nu3 = nu + 0.5 * dt * k2_nu
        quat3 = F.normalize(quat + 0.5 * dt * k2_eta[:, 3:7], dim=-1)
        pos3 = position + 0.5 * dt * k2_eta[:, :3]
        k3_nu, k3_eta = _ode_rhs(nu3, quat3, pos3)
        # k4
        nu4 = nu + dt * k3_nu
        quat4 = F.normalize(quat + dt * k3_eta[:, 3:7], dim=-1)
        pos4 = position + dt * k3_eta[:, :3]
        k4_nu, k4_eta = _ode_rhs(nu4, quat4, pos4)

        nu_dot = (k1_nu + 2 * k2_nu + 2 * k3_nu + k4_nu) / 6.0
        eta_dot = (k1_eta + 2 * k2_eta + 2 * k3_eta + k4_eta) / 6.0

        # NaN 保护: 若 RK4 产出 NaN, 退化为零加速度预测
        nu_dot = torch.nan_to_num(nu_dot, nan=0.0)
        eta_dot = torch.nan_to_num(eta_dot, nan=0.0)

        next_nu = nu + nu_dot * dt
        next_position = position + eta_dot[:, :3] * dt
        next_dq = eta_dot[:, 3:7] * dt
        next_quat = quat + next_dq
        next_quat = F.normalize(next_quat, dim=-1)

        # 净化物理参数 (防止未检测 NaN 通过 L_phys 污染 loss)
        safe_params = {}
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                safe_params[k] = torch.nan_to_num(v, nan=0.0)
            else:
                safe_params[k] = v

        return next_position, next_quat, next_nu, nu_dot, safe_params

    def forward(self, batch, rollout_steps=1):
        """
        前向传播 (支持多步rollout)

        Args:
            batch: dict from DataLoader
            rollout_steps: rollout步数 (1=单步, >1=自回归)

        Returns:
            dict of predictions and intermediate results
        """
        B = batch['position'].shape[0]
        T = batch['position'].shape[1]
        h = self.hist_len

        # ---- 单步预测 (teacher-forcing) ----
        predictions = {
            'nu_dot': [],       # 加速度预测
            'next_nu': [],      # 速度预测
            'next_pos': [],     # 位置预测
            'next_quat': [],    # 姿态预测
        }
        all_params = []

        for t in range(h, T - 1):
            # 当前状态
            pos_t = batch['position'][:, t]
            quat_t = batch['orientation'][:, t]
            nu_t = torch.cat([batch['lin_vel'][:, t], batch['ang_vel'][:, t]], dim=-1)

            # 当前控制/环境
            thr_t = batch['thruster'][:, t]
            dep_t = batch['depth'][:, t]
            alt_t = batch['altitude'][:, t]
            wd_t = batch['wall_dist'][:, t]
            cm_t = batch['contact_mag'][:, t]
            # payload: 无时间维度，取 (B, 1)
            pay_raw = batch['payload']
            if pay_raw.dim() == 2:
                pay_t = pay_raw
            elif pay_raw.dim() == 3:
                pay_t = pay_raw[:, t]
            else:
                pay_t = pay_raw.view(pay_raw.shape[0], 1)

            # 历史窗口
            hist = self._build_hist_input(batch, t - h, t)

            # 前向一步
            next_pos, next_quat, next_nu, nu_dot, params = self.forward_step(
                pos_t, quat_t, nu_t, thr_t, dep_t, alt_t, wd_t, cm_t, pay_t, hist)

            predictions['nu_dot'].append(nu_dot)
            predictions['next_nu'].append(next_nu)
            predictions['next_pos'].append(next_pos)
            predictions['next_quat'].append(next_quat)
            all_params.append(params)

        # Stack
        for k in predictions:
            predictions[k] = torch.stack(predictions[k], dim=1)  # (B, T-h-1, dim)

        # GT targets
        T_pred = T - h - 1
        gt_next_pos = batch['position'][:, h+1:h+1+T_pred]
        gt_next_quat = batch['orientation'][:, h+1:h+1+T_pred]
        gt_next_linvel = batch['lin_vel'][:, h+1:h+1+T_pred]
        gt_next_angvel = batch['ang_vel'][:, h+1:h+1+T_pred]
        gt_next_nu = torch.cat([gt_next_linvel, gt_next_angvel], dim=-1)

        # 速度差分 (归一化目标)
        gt_linvel_diff = batch['lin_vel'][:, h+1:h+1+T_pred] - batch['lin_vel'][:, h:h+T_pred]
        gt_angvel_diff = batch['ang_vel'][:, h+1:h+1+T_pred] - batch['ang_vel'][:, h:h+T_pred]
        gt_pos_diff = batch['position'][:, h+1:h+1+T_pred] - batch['position'][:, h:h+T_pred]

        return {
            'predictions': predictions,
            'gt': {
                'next_pos': gt_next_pos,
                'next_quat': gt_next_quat,
                'next_nu': gt_next_nu,
                'next_linvel': gt_next_linvel,
                'next_angvel': gt_next_angvel,
                'linvel_diff': gt_linvel_diff,
                'angvel_diff': gt_angvel_diff,
                'pos_diff': gt_pos_diff,
            },
            'params': all_params,
            'T_pred': T_pred,
        }

    def rollout(self, batch, n_steps=20, start_t=None):
        """
        自回归 rollout (free-running)

        从 batch 中的某个时刻开始，用模型自身预测作为下一步输入
        """
        B = batch['position'].shape[0]
        T = batch['position'].shape[1]
        h = self.hist_len

        if start_t is None:
            start_t = h + 5  # 留一些历史

        # 初始状态
        pos = batch['position'][:, start_t].clone()
        quat = batch['orientation'][:, start_t].clone()
        lin_vel = batch['lin_vel'][:, start_t].clone()
        ang_vel = batch['ang_vel'][:, start_t].clone()

        # 历史窗口 (用GT填充)
        hist_frames = []
        for t in range(start_t - h, start_t):
            frame = self._build_input(
                batch['position'][:, t], batch['orientation'][:, t],
                batch['lin_vel'][:, t], batch['ang_vel'][:, t],
                batch['thruster'][:, t], batch['depth'][:, t],
                batch['altitude'][:, t], batch['wall_dist'][:, t],
                batch['contact_mag'][:, t],
                batch['payload'][:, t] if batch['payload'].dim() == 3 else batch['payload'][:, 0:1],
            )
            hist_frames.append(frame)
        hist_input = torch.stack(hist_frames, dim=1)  # (B, h, input_dim)

        rollout_pos = [pos.clone()]
        rollout_quat = [quat.clone()]
        rollout_nu = [torch.cat([lin_vel, ang_vel], dim=-1).clone()]

        for step in range(n_steps):
            t_gt = start_t + step
            if t_gt >= T:
                break

            nu = torch.cat([lin_vel, ang_vel], dim=-1)
            thr = batch['thruster'][:, t_gt]
            dep = batch['depth'][:, t_gt]
            alt = batch['altitude'][:, t_gt]
            wd = batch['wall_dist'][:, t_gt]
            cm = batch['contact_mag'][:, t_gt]
            pay = batch['payload'][:, t_gt] if batch['payload'].dim() == 3 else batch['payload'][:, 0:1]

            next_pos, next_quat, next_nu, _, _ = self.forward_step(
                pos, quat, nu, thr, dep, alt, wd, cm, pay, hist_input)

            # 位置裁剪到水池内
            from track_data_loader import POOL_X_MIN, POOL_X_MAX, POOL_Y_MIN, POOL_Y_MAX
            next_pos[:, 0] = next_pos[:, 0].clamp(POOL_X_MIN, POOL_X_MAX)
            next_pos[:, 1] = next_pos[:, 1].clamp(POOL_Y_MIN, POOL_Y_MAX)
            # Z裁剪简化 (依赖y值)

            rollout_pos.append(next_pos.clone())
            rollout_quat.append(next_quat.clone())
            rollout_nu.append(next_nu.clone())

            # 更新状态
            pos = next_pos.detach()
            quat = next_quat.detach()
            lin_vel = next_nu[:, :3].detach()
            ang_vel = next_nu[:, 3:6].detach()

            # 更新历史窗口
            new_frame = self._build_input(
                pos, quat, lin_vel, ang_vel, thr, dep, alt, wd, cm, pay)
            hist_input = torch.cat([hist_input[:, 1:, :], new_frame.unsqueeze(1)], dim=1)

        return {
            'rollout_pos': torch.stack(rollout_pos, dim=1),   # (B, n_steps+1, 3)
            'rollout_quat': torch.stack(rollout_quat, dim=1),
            'rollout_nu': torch.stack(rollout_nu, dim=1),
        }


# ═══════════════════════════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════════════════════════

class GreyBoxLoss(nn.Module):
    """
    灰箱模型训练损失

    L = L_dyn + λ_phys * L_phys + λ_boundary * L_boundary + λ_contact * L_contact

    课程学习:
      Phase 1 (epoch 1-30): 单步预测为主，物理约束权重高
      Phase 2 (epoch 31-60): 引入短rollout
      Phase 3 (epoch 61-100): 长rollout，完整损失
    """
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.w_pos = cfg.get('w_pos', 1.0)
        self.w_quat = cfg.get('w_quat', 0.5)
        self.w_linvel = cfg.get('w_linvel', 2.0)
        self.w_angvel = cfg.get('w_angvel', 1.0)
        self.w_nu_dot = cfg.get('w_nu_dot', 1.0)
        self.w_phys = cfg.get('w_phys', 0.5)
        self.w_boundary = cfg.get('w_boundary', 0.3)
        self.w_contact = cfg.get('w_contact', 0.3)

        # 课程进度
        self.register_buffer('_progress', torch.tensor(0.0), persistent=False)

    def set_progress(self, progress):
        self._progress.fill_(max(0.0, min(1.0, progress)))

    def forward(self, model_output, batch):
        pred = model_output['predictions']
        gt = model_output['gt']
        params_list = model_output['params']
        T_pred = model_output['T_pred']
        h = model_output.get('_h', 10)

        # ---- 1. 动力学预测损失 ----
        L_pos = F.huber_loss(pred['next_pos'], gt['next_pos'], delta=0.1)

        # 速度损失 (核心)
        pred_linvel = pred['next_nu'][:, :, :3]
        pred_angvel = pred['next_nu'][:, :, 3:6]
        L_linvel = F.huber_loss(pred_linvel, gt['next_linvel'], delta=0.05)
        L_angvel = F.huber_loss(pred_angvel, gt['next_angvel'], delta=0.05)

        L_dyn = (self.w_pos * L_pos +
                 self.w_linvel * L_linvel +
                 self.w_angvel * L_angvel)

        # ---- 2. 物理约束损失 ----
        # 仅做轻柔约束: CholeskyHead 已保证对称/正定, 此处只加合理下限
        L_phys = torch.tensor(0.0, device=L_dyn.device)
        n_phys_steps = min(4, len(params_list))

        for params in params_list[:n_phys_steps]:
            MA = params['MA']
            T = params['thruster_matrix']
            DL_vec = params['DL_vec']
            grav = params['gravity']

            # ① 推力器非退化: 每个 6DOF 至少有一个推力器通道响应 > 1e-3
            t_per_dof = T.abs().max(dim=-1)[0]
            L_phys += F.relu(0.001 - t_per_dof).mean()

            # ② 附加质量对角线 ≥ 1e-3
            ma_diag = MA.diagonal(dim1=-2, dim2=-1)
            L_phys += F.relu(0.001 - ma_diag).mean()

            # ③ 阻尼对角线 ≥ 1e-3
            L_phys += F.relu(0.001 - DL_vec).mean()

            # ④ 重浮力差下限
            W = grav[:, 0].abs() + 10.0
            B = grav[:, 1].abs() + 10.0
            L_phys += F.relu(0.1 - (W - B).abs()).mean()

        L_phys = L_phys / n_phys_steps

        # ---- 3. 边界惩罚损失 ----
        wall_dist = batch['wall_dist'][:, h+1:h+1+T_pred]
        boundary_margin = 0.3
        L_boundary = F.relu(boundary_margin - wall_dist).pow(2).mean()

        # ---- 4. 碰撞接触损失 ----
        contact_mag = batch['contact_mag'][:, h+1:h+1+T_pred]
        wall_normals = batch['wall_normals'][:, h+1:h+1+T_pred]
        is_colliding = (contact_mag > 0.1).float()
        if is_colliding.sum() > 0:
            pred_nu_lin = pred['next_nu'][:, :, :3]
            nearest_wall_idx = wall_dist.argmin(dim=-1, keepdim=True)
            nearest_n = torch.gather(wall_normals, -2,
                                     nearest_wall_idx.unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(-2)
            v_normal = (pred_nu_lin * nearest_n).sum(dim=-1)
            L_contact = F.relu(v_normal * is_colliding.squeeze(-1) + 0.05).mean()
        else:
            L_contact = torch.tensor(0.0, device=L_dyn.device)

        # ---- 总损失 ----
        p = float(self._progress)
        # 课程学习: 物理约束权重随进度递减
        phys_scale = max(0.1, 1.0 - p)
        boundary_scale = min(1.0, p * 3)  # 后期加强边界约束
        contact_scale = min(1.0, p * 3)

        L_total = L_dyn + self.w_phys * phys_scale * L_phys + \
                  self.w_boundary * boundary_scale * L_boundary + \
                  self.w_contact * contact_scale * L_contact

        L_total = torch.nan_to_num(L_total, nan=0.0, posinf=10.0, neginf=0.0)

        metrics = {
            'loss_total': L_total.item(),
            'L_dyn': L_dyn.item(),
            'L_pos': L_pos.item(),
            'L_linvel': L_linvel.item(),
            'L_angvel': L_angvel.item(),
            'L_phys': L_phys.item(),
            'L_boundary': L_boundary.item(),
            'L_contact': L_contact.item(),
        }

        return L_total, metrics


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = GreyBoxDynamics()
    print(f"参数量: {count_parameters(model):,}")
    print(f"  PhysicsParamNet: {count_parameters(model.physics_net):,}")
    print(f"  ResidualGRU: {count_parameters(model.residual_gru):,}")

    # 快速测试
    B, T, h = 4, 32, 10
    batch = {
        'position': torch.randn(B, T, 3),
        'orientation': F.normalize(torch.randn(B, T, 4), dim=-1),
        'lin_vel': torch.randn(B, T, 3) * 0.3,
        'ang_vel': torch.randn(B, T, 3) * 0.1,
        'lin_accel': torch.randn(B, T, 3) * 0.5,
        'ang_accel': torch.randn(B, T, 3) * 0.1,
        'thruster': torch.randn(B, T, 8) * 0.5,
        'depth': torch.randn(B, T, 1) * 3,
        'altitude': torch.randn(B, T, 1) * 2,
        'wall_dist': torch.rand(B, T, 6) * 5,
        'wall_normals': torch.randn(B, T, 6, 3),
        'contact_mask': torch.rand(B, T, 6),
        'contact_mag': torch.rand(B, T, 1) * 0.1,
        'payload': torch.zeros(B, 1),
    }

    out = model(batch)
    print(f"\n前向传播成功!")
    print(f"  预测步数: {out['T_pred']}")
    print(f"  nu_dot shape: {out['predictions']['nu_dot'].shape}")
    print(f"  next_pos shape: {out['predictions']['next_pos'].shape}")
