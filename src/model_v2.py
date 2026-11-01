#!/usr/bin/env python3
"""
v2 纯速度空间动力学引擎

设计原则:
  - 输入: 仅用 ROV 真实传感器可观测的量 (IMU, DVL, Pressure, 推力器指令)
  - 输出: 下一帧机体系速度 ν_{t+1} + 加速度 ν̇
  - 不依赖 GT 位姿 (position/orientation) 作为输入
  - 四元数仅用于 Fossen ODE 中的重力补偿 (IMU 可提供)

与 v21 的关键区别:
  ┌────────────────── v21 ──────────────────┐  ┌──────────── v2 ────────────┐
  │ 输入: pos(3) + quat(4) + vel(6) + ...   │  │ 输入: vel(6) + quat(4)     │
  │ 输出: pos(3) + quat(4) + vel(6) + ν̇(6) │  │        + thr + env + hist  │
  │ 监督: 位置 + 速度                       │  │ 输出: vel(6) + ν̇(6)       │
  │ 本质: 轨迹预测                           │  │ 监督: 速度                 │
  └─────────────────────────────────────────┘  │ 本质: 动力学辨识           │
                                                └─────────────────────────────┘

集成到 v20 Dreamer 的方式:
  1. 世界模型 latent state → QuatHead → quaternion (4dim)
  2. 世界模型 latent state → BoundaryHead → wall_dist (6dim)   [Phase 2]
  3. quaternion + velocity + thruster + wall_dist → 本引擎 → next_velocity
  4. next_velocity 作为额外 proprio 特征馈入 RSSM encoder
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    PhysicsParamNet, ResidualGRUNet, CholeskyHead,
    quat_to_rotation_matrix, _skew, construct_coriolis,
    compute_restore_force,
)

TARGET_HZ = 15.0

# ═══════════════════════════════════════════════════════════════════
# Gazebo 真值物理参数 (来源: dlrov_description/urdf/base.xacro + gazebo.xacro)
# ═══════════════════════════════════════════════════════════════════

# 刚体质量/惯量 (不含附加质量, 来源: base.xacro mass=500, inertia)
M_RB_DIAG = torch.tensor([500.0, 500.0, 500.0, 140.26, 259.58, 220.16])

# Gazebo 附加质量矩阵对角线 (gazebo.xacro added_mass, 用于网络初始化)
GAZEBO_MA_DIAG = torch.tensor([160.4, 251.3, 752.6, 38.3, 60.4, 16.1])

# Gazebo 线性阻尼 D_L (gazebo.xacro linDamp, 绝对值)
GAZEBO_DL = torch.tensor([44.2, 41.0, 429.9, 55.3, 63.7, 21.6])

# Gazebo 二次阻尼 D_Q (gazebo.xacro quadDamp, 绝对值)
GAZEBO_DQ = torch.tensor([260.7, 345.9, 634.6, 81.6, 94.1, 63.6])

# 推力器满量程 (snippets.xacro: max_thrust=249.704 N)
THRUST_FULL_SCALE = 250.0

# 最小有效阻尼 (保守安全网, 仅防数值奇异)
# 有效阻尼 d_eff = DL_vec[i] / M_RB[i]
# 50% 速度衰减时间 T50 = ln(2)/d_eff:
#   surge/sway:  d_eff≥0.01 → T50≤69s
#   heave:       d_eff≥0.02 → T50≤35s
#   roll/pitch:  d_eff≥0.03 → T50≤23s
#   yaw:         d_eff≥0.05 → T50≤14s
MIN_EFFECTIVE_DAMPING = torch.tensor([0.01, 0.01, 0.02, 0.03, 0.03, 0.05])


# ═══════════════════════════════════════════════════════════════════
# 辅助物理函数
# ═══════════════════════════════════════════════════════════════════

def compute_repulsive_boundary_force(nu_lin, wall_dist, wall_normals,
                                     margin=0.5, k_repulse=2.0):
    """
    从 6 面墙计算排斥力, 直接加入 ODE rhs

    wall_normals 指向水池内部 (e.g. z_bot normal = +z 向上).
    排斥力方向 = +normal (把 ROV 推离墙面, 即推向池内).

    Args:
        nu_lin:       (batch, 3)   机体系线速度 (世界系)
        wall_dist:    (batch, 6)   到 6 面墙的距离
        wall_normals: (batch, 6, 3) 6 面墙的内法线 (指向水池内部)
        margin:       float        排斥力激活距离 (m)
        k_repulse:    float        排斥力刚度 (N/m)

    Returns:
        F_repulse: (batch, 3) 世界系排斥力
    """
    penetration = F.relu(margin - wall_dist)  # (B, 6), >0 表示太近
    # 排斥力: +k * penetration * normal (推向 pool 内部, 即远离墙)
    F_per_wall = k_repulse * penetration.unsqueeze(-1) * wall_normals  # (B, 6, 3)

    # 速度朝墙时额外阻尼 (防止振荡)
    v_toward_wall = (nu_lin.unsqueeze(1) * wall_normals).sum(dim=-1)  # (B, 6), + = 朝池内
    # 速度朝向墙 = 速度在 -normal 方向有分量, 即 v_toward_wall < 0
    v_into_wall = F.relu(-v_toward_wall)  # (B, 6), >0 表示正在撞墙
    damping_per_wall = k_repulse * v_into_wall * penetration  # 额外阻尼力
    F_damping = damping_per_wall.unsqueeze(-1) * wall_normals  # 沿 normal 方向 (推离墙)

    F_repulse = (F_per_wall + F_damping).sum(dim=1)  # (B, 3)
    return F_repulse


def compute_kinetic_energy(nu, M_raw):
    """
    计算机体系动能: KE = 0.5 * νᵀ · M · ν

    Args:
        nu:    (batch, 6) 机体系速度
        M_raw: (batch, 6, 6) 全惯性矩阵 (M_RB + M_A)

    Returns:
        KE: (batch,) 动能标量
    """
    Mv = torch.bmm(M_raw, nu.unsqueeze(-1)).squeeze(-1)  # (B, 6)
    KE = 0.5 * (nu * Mv).sum(dim=-1)
    return KE


# ═══════════════════════════════════════════════════════════════════
# 纯速度空间 Fossen ODE (改进版)
# ═══════════════════════════════════════════════════════════════════

def fossen_velocity_dynamics(nu, quat, params, thruster_input,
                             wall_dist=None, wall_normals=None,
                             k_wall=8.0, wall_margin=1.0,
                             residual=None, m_rb_diag=None):
    """
    Fossen 6-DOF 纯速度动力学 (不计算位置运动学)

    改进 (v2.1):
      - 壁面排斥力直接加入 ODE rhs, 而非仅靠 loss 惩罚
      - 分线性/角速度独立阻尼尺度
      - 保证每 DOF 最小有效阻尼
      - m_rb_diag: 可选, 兼容旧 checkpoint 的 M_RB 值

    Args:
        nu:             (batch, 6)   机体系速度 [u,v,w,p,q,r]
        quat:           (batch, 4)   四元数 [x,y,z,w] 仅用于重力补偿
        params:         dict          from PhysicsParamNet
        thruster_input: (batch, 8)   推力器指令
        wall_dist:      (batch, 6)   壁面距离 (可选, 用于排斥力)
        wall_normals:   (batch, 6, 3) 壁面法线 (可选)
        k_wall:         float        排斥力刚度
        wall_margin:    float        排斥力激活距离
        residual:       (batch, 6)   残差加速度

    Returns:
        nu_dot: (batch, 6) 机体系加速度
    """
    MA = params['MA']
    DL_vec = params['DL_vec']
    DQ_vec = params['DQ_vec']
    T = params['thruster_matrix']

    # 使用传入的 M_RB 或模块默认值 (兼容旧 checkpoint)
    m_rb = m_rb_diag if m_rb_diag is not None else M_RB_DIAG
    m_scale = m_rb.to(device=nu.device, dtype=nu.dtype)
    M_RB = torch.diag_embed(m_scale).unsqueeze(0).expand(nu.shape[0], -1, -1)
    M_raw = M_RB + MA

    inv_scale = 1.0 / m_scale
    M = M_raw * inv_scale.unsqueeze(-1)

    C_raw = construct_coriolis(M_raw, nu)
    C = C_raw * inv_scale.unsqueeze(-1)

    # ── 阻尼: 保证最小有效阻尼 ──
    # 有效阻尼 d_eff[i] = DL_vec[i] / m_scale[i]
    # 要求 d_eff[i] >= MIN_EFFECTIVE_DAMPING[i]
    min_eff = MIN_EFFECTIVE_DAMPING.to(device=nu.device, dtype=nu.dtype)
    min_dl = min_eff * m_scale  # DL_vec 的最小值 (除以 m_scale 前)
    DL_vec_eff = DL_vec + F.relu(min_dl - DL_vec)  # 补齐到最小值以上
    DQ_vec_eff = DQ_vec + F.relu(0.5 * min_dl - DQ_vec)  # 二次阻尼也设下限

    damping_raw = DL_vec_eff * nu + DQ_vec_eff * nu.abs() * nu
    damping = damping_raw * inv_scale

    g_force_raw = compute_restore_force(quat, params['gravity'])
    g_force = g_force_raw * inv_scale

    tau_raw = torch.bmm(T, thruster_input.unsqueeze(-1)).squeeze(-1)
    tau = tau_raw * inv_scale

    # ── 壁面排斥力 (直接加入 ODE rhs) ──
    F_boundary = torch.zeros_like(tau[:, :3])
    if wall_dist is not None and wall_normals is not None:
        F_repulse = compute_repulsive_boundary_force(
            nu[:, :3], wall_dist, wall_normals, margin=wall_margin, k_repulse=k_wall)
        F_boundary = F_repulse * inv_scale[:3]  # 同样做质量归一化

    rhs_lin = tau[:, :3] - torch.bmm(C[:, :3, :], nu.unsqueeze(-1)).squeeze(-1) \
              - damping[:, :3] - g_force[:, :3] + F_boundary
    rhs_ang = tau[:, 3:6] - torch.bmm(C[:, 3:6, :], nu.unsqueeze(-1)).squeeze(-1) \
              - damping[:, 3:6] - g_force[:, 3:6]
    rhs = torch.cat([rhs_lin, rhs_ang], dim=-1)
    rhs = torch.clamp(rhs, -50.0, 50.0)

    # 正则化 M 防奇异: 多重 fallback
    eye6 = torch.eye(6, device=M.device, dtype=M.dtype).unsqueeze(0)
    nu_dot_phys = None
    for eps in [1e-3, 1e-1, 1.0, 10.0]:
        try:
            M_reg = M + eps * eye6
            nu_dot_phys = torch.linalg.solve(M_reg, rhs.unsqueeze(-1)).squeeze(-1)
            break
        except Exception:
            continue

    if nu_dot_phys is None:
        # 最终 fallback: 仅用 M 对角线求近似逆 (保证不崩溃)
        m_diag = M.diagonal(dim1=-2, dim2=-1).abs().clamp(min=1e-3)  # (B, 6)
        nu_dot_phys = rhs / m_diag
    nu_dot_phys = torch.clamp(nu_dot_phys, -5.0, 5.0)
    nu_dot_phys = torch.nan_to_num(nu_dot_phys, nan=0.0, posinf=5.0, neginf=-5.0)

    if residual is not None:
        residual = torch.clamp(residual, -3.0, 3.0)
        nu_dot = nu_dot_phys + residual
    else:
        nu_dot = nu_dot_phys

    return nu_dot


# ═══════════════════════════════════════════════════════════════════
# VelocityDynamics 主模型
# ═══════════════════════════════════════════════════════════════════

class VelocityDynamics(nn.Module):
    """
    纯速度空间动力学引擎

    输入/输出都不包含绝对位置。四元数仅作为重力补偿的辅助输入，
    模型不预测下一帧四元数。
    """
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}

        self.linvel_dim = 3
        self.angvel_dim = 3
        self.quat_dim = 4
        self.total_vel_dim = 6
        self.accel_proxy_dim = 3

        self.thruster_dim = cfg.get('thruster_dim', 8)
        self.env_dim = cfg.get('env_dim', 9)
        self.payload_dim = cfg.get('payload_dim', 1)
        self.hist_len = cfg.get('hist_len', 10)
        self.jump_frame = cfg.get('jump_frame', 0)
        self.target_hz = cfg.get('target_hz', TARGET_HZ)
        self.damping_scale = cfg.get('damping_scale', 10.0)

        # ── M_RB_DIAG: 可从 config 覆盖, 兼容旧 checkpoint ──
        m_rb_cfg = cfg.get('m_rb_diag', None)
        if m_rb_cfg is not None:
            self.register_buffer('_m_rb_diag', torch.tensor(m_rb_cfg, dtype=torch.float32), persistent=False)
        else:
            self.register_buffer('_m_rb_diag', M_RB_DIAG.clone(), persistent=False)

        # ── v2.1 改进: 独立线性/角速度阻尼尺度 ──
        self.damping_scale_linear = cfg.get('damping_scale_linear', self.damping_scale)
        self.damping_scale_angular = cfg.get('damping_scale_angular', self.damping_scale * 3.0)
        # ── v2.1 改进: 壁面排斥力参数 (软化: 边界检测不精确, 机械爪先于本体碰撞) ──
        self.k_wall = cfg.get('k_wall', 2.0)           # 排斥力刚度 (软弹簧)
        self.wall_margin = cfg.get('wall_margin', 0.5)  # 排斥力激活距离 (m), 仅很近时触发
        # ── v2.1 改进: 推力器死区阈值 ──
        self.thrust_deadzone = cfg.get('thrust_deadzone', 0.02)  # 归一化死区 (相对满推力)
        self.thrust_full_scale = cfg.get('thrust_full_scale', THRUST_FULL_SCALE)  # 满推力值 (Gazebo: 250N)

        self.input_dim = (
            self.linvel_dim + self.angvel_dim + self.quat_dim +
            self.thruster_dim + self.env_dim + self.accel_proxy_dim + self.payload_dim
        )

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

        self.register_buffer('_norm_mean', torch.zeros(self.input_dim), persistent=False)
        self.register_buffer('_norm_std', torch.ones(self.input_dim), persistent=False)

    def set_norm_stats(self, mean, std):
        self._norm_mean.copy_(mean)
        self._norm_std.copy_(std)

    def _normalize(self, x):
        return (x - self._norm_mean) / self._norm_std.clamp(min=1e-4)

    def _build_input(self, lin_vel, ang_vel, quat, thruster, depth, altitude,
                     wall_dist, contact_mag, accel_proxy, payload):
        raw = torch.cat([
            lin_vel, ang_vel, quat,
            thruster, depth, altitude, wall_dist, contact_mag, accel_proxy, payload
        ], dim=-1)
        return self._normalize(raw)

    def _build_hist_input(self, batch, t_start, t_end):
        B = batch['lin_vel'].shape[0]
        h = t_end - t_start

        lv = batch['lin_vel'][:, t_start:t_end]
        av = batch['ang_vel'][:, t_start:t_end]
        qu = batch['orientation'][:, t_start:t_end]
        thr = batch['thruster'][:, t_start:t_end]
        dep = batch['depth'][:, t_start:t_end]
        alt = batch['altitude'][:, t_start:t_end]
        wd = batch['wall_dist'][:, t_start:t_end]
        cm = batch['contact_mag'][:, t_start:t_end]
        ap = batch['lin_accel'][:, t_start:t_end]

        pay_raw = batch['payload']
        if pay_raw.dim() == 2:
            pay = pay_raw.unsqueeze(1).expand(B, h, 1)
        elif pay_raw.dim() == 3:
            pay = pay_raw[:, t_start:t_end]
        else:
            pay = pay_raw.view(B, 1).unsqueeze(1).expand(B, h, 1)

        raw = torch.cat([lv, av, qu, thr, dep, alt, wd, cm, ap[:, :, :3], pay], dim=-1)
        return self._normalize(raw)

    @staticmethod
    def quat_derivative(quat, ang_vel):
        omega = ang_vel
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        dq = torch.stack([
            0.5 * (qw * omega[:, 0] + qz * omega[:, 1] - qy * omega[:, 2]),
            0.5 * (-qz * omega[:, 0] + qw * omega[:, 1] + qx * omega[:, 2]),
            0.5 * (qy * omega[:, 0] - qx * omega[:, 1] + qw * omega[:, 2]),
            0.5 * (-qx * omega[:, 0] - qy * omega[:, 1] - qz * omega[:, 2]),
        ], dim=-1)
        return dq

    @torch.no_grad()
    def integrate_quat(self, quat, ang_vel, dt):
        return F.normalize(quat + self.quat_derivative(quat, ang_vel) * dt, dim=-1)

    def forward_step(self, lin_vel, ang_vel, quat, thruster_input,
                     depth, altitude, wall_dist, wall_normals, contact_mag, accel_proxy,
                     payload, hist_input):
        """
        单步前向: velocity → next velocity

        Args:
            lin_vel:        (batch, 3)
            ang_vel:        (batch, 3)
            quat:           (batch, 4)  仅用于重力补偿
            thruster_input: (batch, 8)  ODE 推力均值
            depth:          (batch, 1)
            altitude:       (batch, 1)
            wall_dist:      (batch, 6)
            wall_normals:   (batch, 6, 3)  壁面内法线 (v2.1 新增)
            contact_mag:    (batch, 1)
            accel_proxy:    (batch, 3)  IMU线加速度代理
            payload:        (batch, 1)
            hist_input:     (batch, h, input_dim)

        Returns:
            next_nu:  (batch, 6)  预测下一帧速度
            nu_dot:   (batch, 6)  预测加速度
            params:   dict        物理参数
            residual: (batch, 6)  残差加速度
        """
        nu = torch.cat([lin_vel, ang_vel], dim=-1)

        params = self.physics_net(hist_input, payload)
        # 线性/角速度独立阻尼尺度
        params['DL_vec'] = params['DL_vec'].clone()
        DQ_vec_scaled = params['DQ_vec'].clone()
        params['DL_vec'][:, :3] = params['DL_vec'][:, :3] * self.damping_scale_linear
        params['DL_vec'][:, 3:6] = params['DL_vec'][:, 3:6] * self.damping_scale_angular
        DQ_vec_scaled[:, :3] = DQ_vec_scaled[:, :3] * self.damping_scale_linear
        DQ_vec_scaled[:, 3:6] = DQ_vec_scaled[:, 3:6] * self.damping_scale_angular
        params['DQ_vec'] = DQ_vec_scaled
        residual = self.residual_gru(hist_input)

        # ── 推力器 ODE 均值 (处理 jump_frame 展开) ──
        thr_ode = thruster_input
        if self.thruster_dim > 8 and thruster_input.shape[-1] == self.thruster_dim:
            dt_mult = self.jump_frame + 1
            thr_ode = thruster_input.view(-1, dt_mult, 8).mean(dim=1)

        # ── 推力器死区: 小指令置零 (模拟真实推力器死区) ──
        thr_norm = thr_ode.abs().max(dim=-1, keepdim=True)[0]
        deadzone_mask = (thr_norm < self.thrust_deadzone * self.thrust_full_scale).float()
        thr_ode_clean = thr_ode * (1.0 - deadzone_mask)

        # ── 壁面法线安全处理 ──
        wd_t = wall_dist
        wn_t = wall_normals if wall_normals is not None else torch.zeros(
            nu.shape[0], 6, 3, device=nu.device, dtype=nu.dtype)

        # 首次 ODE 求值 (k1)
        nu_dot = fossen_velocity_dynamics(
            nu, quat, params, thr_ode_clean,
            wall_dist=wd_t, wall_normals=wn_t,
            k_wall=self.k_wall, wall_margin=self.wall_margin,
            residual=residual, m_rb_diag=self._m_rb_diag)

        dt = (self.jump_frame + 1) / self.target_hz

        k1_nu = nu_dot
        k1_q = self.quat_derivative(quat, nu[:, 3:6])

        nu2 = nu + 0.5 * dt * k1_nu
        quat2 = F.normalize(quat + 0.5 * dt * k1_q, dim=-1)
        k2_nu = fossen_velocity_dynamics(nu2, quat2, params, thr_ode_clean,
                                         wall_dist=wd_t, wall_normals=wn_t,
                                         k_wall=self.k_wall, wall_margin=self.wall_margin,
                                         residual=None, m_rb_diag=self._m_rb_diag)
        k2_q = self.quat_derivative(quat2, nu2[:, 3:6])

        nu3 = nu + 0.5 * dt * k2_nu
        quat3 = F.normalize(quat + 0.5 * dt * k2_q, dim=-1)
        k3_nu = fossen_velocity_dynamics(nu3, quat3, params, thr_ode_clean,
                                         wall_dist=wd_t, wall_normals=wn_t,
                                         k_wall=self.k_wall, wall_margin=self.wall_margin,
                                         residual=None, m_rb_diag=self._m_rb_diag)
        k3_q = self.quat_derivative(quat3, nu3[:, 3:6])

        nu4 = nu + dt * k3_nu
        quat4 = F.normalize(quat + dt * k3_q, dim=-1)
        k4_nu = fossen_velocity_dynamics(nu4, quat4, params, thr_ode_clean,
                                         wall_dist=wd_t, wall_normals=wn_t,
                                         k_wall=self.k_wall, wall_margin=self.wall_margin,
                                         residual=None, m_rb_diag=self._m_rb_diag)

        nu_dot_rk4 = (k1_nu + 2 * k2_nu + 2 * k3_nu + k4_nu) / 6.0
        nu_dot_rk4 = torch.nan_to_num(nu_dot_rk4, nan=0.0)

        next_nu = nu + nu_dot_rk4 * dt

        safe_params = {}
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                safe_params[k] = torch.nan_to_num(v, nan=0.0)
            else:
                safe_params[k] = v

        return next_nu, nu_dot_rk4, safe_params, residual

    def forward(self, batch):
        B = batch['lin_vel'].shape[0]
        T = batch['lin_vel'].shape[1]
        h = self.hist_len

        predictions = {'nu_dot': [], 'next_nu': []}
        all_params = []
        all_residuals = []

        for t in range(h, T - 1):
            lv_t = batch['lin_vel'][:, t]
            av_t = batch['ang_vel'][:, t]
            qu_t = batch['orientation'][:, t]
            thr_t = batch['thruster'][:, t]
            dep_t = batch['depth'][:, t]
            alt_t = batch['altitude'][:, t]
            wd_t = batch['wall_dist'][:, t]
            wn_t = batch['wall_normals'][:, t]
            cm_t = batch['contact_mag'][:, t]
            ap_t = batch['lin_accel'][:, t]

            pay_raw = batch['payload']
            if pay_raw.dim() == 2:
                pay_t = pay_raw
            elif pay_raw.dim() == 3:
                pay_t = pay_raw[:, t]
            else:
                pay_t = pay_raw.view(pay_raw.shape[0], 1)

            hist = self._build_hist_input(batch, t - h, t)
            next_nu, nu_dot, params, residual = self.forward_step(
                lv_t, av_t, qu_t, thr_t, dep_t, alt_t, wd_t, wn_t, cm_t, ap_t, pay_t, hist)

            predictions['nu_dot'].append(nu_dot)
            predictions['next_nu'].append(next_nu)
            all_params.append(params)
            all_residuals.append(residual)

        for k in predictions:
            predictions[k] = torch.stack(predictions[k], dim=1)

        T_pred = T - h - 1
        gt_next_linvel = batch['lin_vel'][:, h + 1:h + 1 + T_pred]
        gt_next_angvel = batch['ang_vel'][:, h + 1:h + 1 + T_pred]
        gt_next_nu = torch.cat([gt_next_linvel, gt_next_angvel], dim=-1)

        return {
            'predictions': predictions,
            'gt': {
                'next_nu': gt_next_nu,
                'next_linvel': gt_next_linvel,
                'next_angvel': gt_next_angvel,
            },
            'params': all_params,
            'residuals': all_residuals,
            'T_pred': T_pred,
        }

    def forward_rollout(self, batch, n_rollout_steps):
        """
        多步自回归 rollout + 轨迹积分
        
        将预测的速度通过四元数积分到世界系位移,
        输出轨迹位置 (而非速度) 与 GT 位置的对比损失
        
        Returns:
            pred_traj:  (B, n, 3)  积分得到的相对位移轨迹
            gt_traj:    (B, n, 3)  GT 位置轨迹 (相对于起点)
            vel_pred:   (B, n, 6)  预测速度 (用于监控)
            vel_gt:     (B, n, 6)  GT 速度 (用于监控)
        """
        B = batch['lin_vel'].shape[0]
        T = batch['lin_vel'].shape[1]
        h = self.hist_len
        n = min(n_rollout_steps, T - h - 1)
        if n <= 0:
            return None

        dt = (self.jump_frame + 1) / self.target_hz
        pred_vel_list = []
        pred_pos_list = []
        
        # 初始状态: teacher-forcing 状态 (t = h)
        lv_t = batch['lin_vel'][:, h].clone()
        av_t = batch['ang_vel'][:, h].clone()
        qu_t = batch['orientation'][:, h].clone()
        
        # 初始位置 (世界系, 作为积分起点)
        pos_t = batch['position'][:, h].clone()
        
        # 初始历史窗口: frames [0, h)
        hist_input = self._build_hist_input(batch, 0, h)

        for step in range(n):
            t = h + step
            thr_t = batch['thruster'][:, t]
            dep_t = batch['depth'][:, t]
            alt_t = batch['altitude'][:, t]
            wd_t = batch['wall_dist'][:, t]
            wn_t = batch['wall_normals'][:, t]
            cm_t = batch['contact_mag'][:, t]
            ap_t = batch['lin_accel'][:, t]

            pay_raw = batch['payload']
            if pay_raw.dim() == 2:
                pay_t = pay_raw
            elif pay_raw.dim() == 3:
                pay_t = pay_raw[:, t]
            else:
                pay_t = pay_raw.view(pay_raw.shape[0], 1)

            next_nu, nu_dot, _, _ = self.forward_step(
                lv_t, av_t, qu_t, thr_t, dep_t, alt_t, wd_t, wn_t, cm_t, ap_t, pay_t, hist_input)
            
            pred_vel_list.append(next_nu)
            
            # 轨迹积分: body 速度 → world 速度 → 位置累加
            lv_pred = next_nu[:, :3]
            vel_world = quat_rotate(qu_t, lv_pred)  # (B, 3)
            pos_t = pos_t + vel_world * dt
            pred_pos_list.append(pos_t.clone())
            
            # 自回归更新
            lv_t = lv_pred.detach()
            av_t = next_nu[:, 3:6].detach()
            qu_t = self.integrate_quat(qu_t, av_t, dt)
            
            new_frame = self._build_input(lv_t, av_t, qu_t, thr_t, dep_t, alt_t, wd_t, cm_t, ap_t, pay_t)
            hist_input = torch.cat([hist_input[:, 1:, :], new_frame.unsqueeze(1)], dim=1)

        # 预测轨迹 (相对起点)
        pred_traj = torch.stack(pred_pos_list, dim=1)  # (B, n, 3)
        # GT 轨迹 (相对起点)
        gt_traj = batch['position'][:, h + 1:h + 1 + n]  # (B, n, 3)
        
        # 速度 (用于监控)
        vel_pred = torch.stack(pred_vel_list, dim=1)  # (B, n, 6)
        vel_gt = torch.cat([
            batch['lin_vel'][:, h + 1:h + 1 + n],
            batch['ang_vel'][:, h + 1:h + 1 + n],
        ], dim=-1)

        return {
            'pred_traj': pred_traj,
            'gt_traj': gt_traj,
            'vel_pred': vel_pred,
            'vel_gt': vel_gt,
        }

    @torch.no_grad()
    def simulate(self, init_linvel, init_angvel, init_quat,
                 thruster_sequence, depth, altitude, wall_dist, contact_mag,
                 accel_proxy, payload, n_steps, hist_init_fn):
        """
        自回归仿真 (无 GT 信息)

        Args:
            init_linvel:          np.array (3,)
            init_angvel:          np.array (3,)
            init_quat:            np.array (4,)
            thruster_sequence:    np.array (T, 8)
            depth, altitude, ...: scalar or np.array
            n_steps:              int
            hist_init_fn:         callable → 初始历史窗口 (B, h, input_dim)

        Returns:
            dict with keys: lin_vel, ang_vel, nu_dot
        """
        device = next(self.parameters()).device
        to_t = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)

        lv = to_t(init_linvel)
        av = to_t(init_angvel)
        qu = to_t(init_quat)

        hist_input = hist_init_fn(self, lv, av, qu)

        lin_vels = [lv.cpu().numpy()[0].copy()]
        ang_vels = [av.cpu().numpy()[0].copy()]
        nu_dots = []

        for step in range(n_steps):
            thr_raw = to_t(thruster_sequence[step % len(thruster_sequence)])
            thr_expanded = thr_raw.repeat(1, self.jump_frame + 1) if self.jump_frame > 0 else thr_raw
            if isinstance(depth, (int, float)):
                dep = torch.full((1, 1), float(depth), device=device)
            else:
                dep = to_t(depth[step])
            if isinstance(altitude, (int, float)):
                alt = torch.full((1, 1), float(altitude), device=device)
            else:
                alt = to_t(altitude[step])
            wd = to_t(wall_dist[step]) if not isinstance(wall_dist, (int, float)) else \
                 torch.full((1, 6), float(wall_dist), device=device)
            # 模拟器不自带 wall_normals, 生成默认法线 (6面墙 x 3方向)
            wn = torch.tensor([[[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,-1.],[0.,0.,1.]]],
                              device=device, dtype=torch.float32)
            cm = to_t(contact_mag[step]) if not isinstance(contact_mag, (int, float)) else \
                 torch.zeros(1, 1, device=device)
            ap = to_t(accel_proxy[step]) if not isinstance(accel_proxy, (int, float)) else \
                 torch.zeros(1, 3, device=device)
            pay = torch.zeros(1, 1, device=device) if isinstance(payload, (int, float)) and payload == 0 else \
                  to_t(payload)

            next_nu, nu_dot, _, _ = self.forward_step(
                lv, av, qu, thr_expanded, dep, alt, wd, wn, cm, ap, pay, hist_input)

            next_nu = next_nu.detach()
            nu_dot = nu_dot.detach()

            lin_vels.append(next_nu[:, :3].cpu().numpy()[0].copy())
            ang_vels.append(next_nu[:, 3:6].cpu().numpy()[0].copy())
            nu_dots.append(nu_dot.cpu().numpy()[0].copy())

            lv = next_nu[:, :3]
            av = next_nu[:, 3:6]
            dt = (self.jump_frame + 1) / self.target_hz
            qu = self.integrate_quat(qu, av, dt)

            ap_self = nu_dot[:, :3].detach()
            new_frame = self._build_input(lv, av, qu, thr_expanded, dep, alt, wd, cm, ap_self, pay)
            hist_input = torch.cat([hist_input[:, 1:, :], new_frame.unsqueeze(1)], dim=1)

        return {
            'lin_vel': np.stack(lin_vels),
            'ang_vel': np.stack(ang_vels),
            'nu_dot': np.stack(nu_dots) if nu_dots else np.zeros((0, 6)),
        }


# ═══════════════════════════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════════════════════════

class V2Loss(nn.Module):
    """
    v2.1 速度动力学损失 (改进版)

    新增损失项 (v2.1):
      - L_energy: 推力为零时动能必须衰减 (解决 "不会滑行停止")
      - L_yaw_damp: 偏航阻尼专项加强 (解决 "180° 掉头")
      - L_deadzone: 零推力长时间后速度应趋零 (解决 "停不下来")

    改进的训练调度:
      - rollout 从 epoch 5 开始 (原 epoch 2), 更渐进
      - 物理约束不再是均匀衰减, 而是保证最小有效阻尼
      - 边界排斥力已在 ODE 中生效, L_boundary 权重降低

    L = L_dyn + λ_rollout·α·L_rollout + λ_phys·L_phys + λ_energy·L_energy
        + λ_yaw·L_yaw_damp + λ_dead·L_deadzone + λ_res·L_residual
        + λ_bnd·L_boundary + λ_contact·L_contact
    """
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.w_linvel = cfg.get('w_linvel', 2.0)
        self.w_angvel = cfg.get('w_angvel', 1.0)
        self.w_phys = cfg.get('w_phys', 1.0)
        self.w_residual = cfg.get('w_residual', 0.05)
        self.w_boundary = cfg.get('w_boundary', 0.3)
        self.w_contact = cfg.get('w_contact', 0.3)
        self.w_rollout = cfg.get('w_rollout', 3.0)
        # ── v2.1 新权重 ──
        self.w_energy = cfg.get('w_energy', 1.5)       # 能量衰减约束
        self.w_yaw_damp = cfg.get('w_yaw_damp', 2.0)   # 偏航阻尼加强
        self.w_deadzone = cfg.get('w_deadzone', 0.5)   # 死区/静摩擦
        # ── v2.1 改进调度 ──
        self.rollout_steps = cfg.get('rollout_steps', [5, 10, 30])  # 50→30
        self.rollout_start_epoch = cfg.get('rollout_start_epoch', 5)  # 2→5
        self.register_buffer('_progress', torch.tensor(0.0), persistent=False)
        self.register_buffer('_epoch', torch.tensor(0.0), persistent=False)
        self._model = None

    def set_progress(self, progress):
        self._progress.fill_(max(0.0, min(1.0, progress)))

    def set_epoch(self, epoch):
        self._epoch.fill_(float(epoch))

    def set_model(self, model):
        self._model = model

    def forward(self, model_output, batch):
        pred = model_output['predictions']
        gt = model_output['gt']
        params_list = model_output['params']
        residuals_list = model_output.get('residuals', [])
        T_pred = model_output['T_pred']
        h = model_output.get('_h', 10)

        pred_linvel = pred['next_nu'][:, :, :3]
        pred_angvel = pred['next_nu'][:, :, 3:6]
        L_linvel = F.huber_loss(pred_linvel, gt['next_linvel'], delta=0.02)
        L_angvel = F.huber_loss(pred_angvel, gt['next_angvel'], delta=0.02)

        L_dyn = self.w_linvel * L_linvel + self.w_angvel * L_angvel

        # ── Rollout 损失 (改进调度: epoch 5 起, 更渐进) ──
        L_rollout = torch.tensor(0.0, device=L_dyn.device)
        L_rt5 = torch.tensor(0.0, device=L_dyn.device)
        L_rt10 = torch.tensor(0.0, device=L_dyn.device)
        L_rt30 = torch.tensor(0.0, device=L_dyn.device)

        epoch = float(self._epoch)
        rollout_alpha = 0.0
        if epoch >= self.rollout_start_epoch and self._model is not None:
            # 更长的渐进期: 6 epoch 内从 0 到 1
            rollout_alpha = min(1.0, (epoch - self.rollout_start_epoch + 1) / 6.0)

            for n_steps in self.rollout_steps:
                roll = self._model.forward_rollout(batch, n_steps)
                if roll is None:
                    continue
                # 轨迹损失: 预测位置 vs GT 位置 (Huber, delta=0.5m)
                roll_loss = F.huber_loss(roll['pred_traj'], roll['gt_traj'], delta=0.5)
                L_rollout += roll_loss
                if n_steps == 5:
                    L_rt5 = roll_loss.detach()
                elif n_steps == 10:
                    L_rt10 = roll_loss.detach()
                elif n_steps == 30:
                    L_rt30 = roll_loss.detach()
                else:
                    # 处理任意步数
                    pass

            L_rollout = L_rollout / len(self.rollout_steps)

        # ── 物理约束 (改进: per-DOF 最小有效阻尼) ──
        L_phys = torch.tensor(0.0, device=L_dyn.device)
        n_phys_steps = min(4, len(params_list))
        for params in params_list[:n_phys_steps]:
            MA = params['MA']
            T = params['thruster_matrix']
            DL_vec = params['DL_vec']
            DQ_vec = params['DQ_vec']
            grav = params['gravity']

            t_per_dof = T.abs().max(dim=-1)[0]
            L_phys += F.relu(0.001 - t_per_dof).mean() * 0.1
            ma_diag = MA.diagonal(dim1=-2, dim2=-1)
            L_phys += F.relu(0.001 - ma_diag).mean() * 0.1
            # ── 阻尼正定性: ODE 中已有硬安全网, loss 仅做轻柔下限检查 ──
            # DL_vec 在 forward_step 中被 damping_scale 放大, 此处检查缩放后的值 > 0
            L_phys += F.relu(0.01 - DL_vec).mean() * 0.01
            L_phys += F.relu(0.01 - DQ_vec[:, 3:6]).mean() * 0.01
            # 重力/浮力: 不预设正负浮力, 仅约束不要极度失衡
            # Gazebo 真值: W=4905N, B=5246N → 正浮力 341N
            W = grav[:, 0].abs() + 10.0
            B = grav[:, 1].abs() + 10.0
            imbalance = (W - B).abs()
            L_phys += F.relu(imbalance - 2000.0).mean() * 0.01  # 仅约束不要超过 ~2000N 失衡
        L_phys = L_phys / n_phys_steps

        # ── v2.1 能量衰减损失: 零推力时动能必须减小 ──
        # 核心思路: 当推力为零时, 预测的下一步动能不能超过当前动能
        # 这直接解决 "滑行不停" 和 "180° 掉头" 问题
        L_energy = torch.tensor(0.0, device=L_dyn.device)
        if n_phys_steps > 0:
            # 使用第一个预测步的物理参数计算 M_raw
            m_scale = M_RB_DIAG.to(device=L_dyn.device)
            MA_0 = params_list[0]['MA']
            M_raw = torch.diag_embed(m_scale).unsqueeze(0).expand(MA_0.shape[0], -1, -1) + MA_0

            # 当前速度
            nu_curr = torch.cat([
                batch['lin_vel'][:, h:h + T_pred],
                batch['ang_vel'][:, h:h + T_pred]
            ], dim=-1)  # (B, T_pred, 6)
            # 预测速度
            nu_next = pred['next_nu']  # (B, T_pred, 6)

            # 动能: KE = 0.5 * νᵀ · M · ν (简化: 用 M_raw 对角近似)
            m_diag = M_raw.diagonal(dim1=-2, dim2=-1).unsqueeze(1)  # (B, 1, 6)
            KE_curr = 0.5 * (nu_curr * nu_curr * m_diag).sum(dim=-1)  # (B, T_pred)
            KE_next = 0.5 * (nu_next * nu_next * m_diag).sum(dim=-1)  # (B, T_pred)

            # 检测零推力段: thruster 全为零
            thr_t = batch['thruster'][:, h:h + T_pred]  # (B, T_pred, thr_dim)
            thr_mag = thr_t.abs().max(dim=-1)[0]  # (B, T_pred)
            is_idle = (thr_mag < 1.0).float()  # 推力 < 1% 满量程视为零推力

            if is_idle.sum() > 0:
                # 零推力时动能必须减小 (允许很小的容差)
                L_energy = (F.relu(KE_next - KE_curr + 0.001) * is_idle).sum() / (is_idle.sum() + 1)

        # ── v2.1 偏航阻尼专项约束: 直接防止角速度自旋 ──
        L_yaw_damp = torch.tensor(0.0, device=L_dyn.device)
        if n_phys_steps > 0:
            # 检查零推力时预测的角速度是否异常
            nu_next_ang = pred['next_nu'][:, :, 3:6]  # (B, T_pred, 3) [p, q, r]
            thr_t = batch['thruster'][:, h:h + T_pred]
            thr_mag = thr_t.abs().max(dim=-1)[0]
            is_idle = (thr_mag < 1.0).float()
            if is_idle.sum() > 0:
                # 零推力时预测角速度的范数应趋近于零
                ang_norm = nu_next_ang.norm(dim=-1)  # (B, T_pred)
                L_yaw_damp = (ang_norm * is_idle).sum() / (is_idle.sum() + 1)

        # ── v2.1 死区/静摩擦: 长时间零推力后速度应为零 ──
        L_deadzone = torch.tensor(0.0, device=L_dyn.device)
        if T_pred >= 5:
            # 检查连续 5 帧零推力的段
            thr_seq = batch['thruster'][:, h:h + T_pred]
            thr_mag = thr_seq.abs().max(dim=-1)[0]  # (B, T_pred)
            # 滑动窗口: 连续 N 帧零推力 → 第 N 帧速度应接近零
            idle_window = 5
            for offset in range(min(3, T_pred - idle_window)):
                # 检查 [offset, offset+idle_window) 是否全零推力
                consecutive_idle = (thr_mag[:, offset:offset + idle_window] < 1.0).all(dim=-1)
                if consecutive_idle.sum() > 0:
                    # 窗口末尾速度应趋于零
                    vel_at_end = pred['next_nu'][:, offset + idle_window - 1, :]  # (B, 6)
                    vel_norm = vel_at_end.norm(dim=-1)  # (B,)
                    L_deadzone += (vel_norm * consecutive_idle.float()).sum() / (consecutive_idle.sum() + 1)
            L_deadzone = L_deadzone / 3  # 平均

        # ── 残差正则 (改进: 零推力时残差更受限) ──
        L_residual = torch.tensor(0.0, device=L_dyn.device)
        if residuals_list:
            # 取前 min(4, T_pred) 步的残差
            n_res = min(4, len(residuals_list))
            res_norms = torch.stack([r.norm(dim=-1).mean() for r in residuals_list[:n_res]])
            L_residual = res_norms.mean()
            # 零推力时残差应额外受限 (残差不应在滑行时产生驱动力)
            if n_phys_steps > 0 and T_pred >= 2:
                thr_any = batch['thruster'][:, h:h + T_pred].abs().max(dim=-1)[0]  # (B, T_pred)
                idle_mask = (thr_any < 1.0).float()  # (B, T_pred)
                if idle_mask.sum() > 0:
                    # 残差按时间步对齐: residuals_list[t] shape=(B,6)
                    n_res_idle = min(n_res, idle_mask.shape[1])
                    res_idle_sum = 0.0
                    for t in range(n_res_idle):
                        r = residuals_list[t]  # (B, 6)
                        r_norm = r.norm(dim=-1)  # (B,)
                        # idle_mask[:, t] shape (B,)
                        res_idle_sum += (r_norm * idle_mask[:, t]).mean()
                    L_residual += (res_idle_sum / max(1, n_res_idle)) * 2.0

        # ── 边界损失 (权重降低: 排斥力已在 ODE 中生效) ──
        wall_dist = batch['wall_dist'][:, h + 1:h + 1 + T_pred]
        wall_normals = batch['wall_normals'][:, h + 1:h + 1 + T_pred]
        pred_nu_lin = pred['next_nu'][:, :, :3]

        nearest_wall_idx = wall_dist.argmin(dim=-1, keepdim=True)
        nearest_dist = wall_dist.min(dim=-1)[0]
        nearest_n = torch.gather(wall_normals, -2,
                                 nearest_wall_idx.unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(-2)
        v_toward_wall = (pred_nu_lin * nearest_n).sum(dim=-1)
        proximity = F.relu(1.5 - nearest_dist)
        L_boundary = (F.relu(v_toward_wall) * proximity).mean()

        # ── 碰撞损失 ──
        contact_mag = batch['contact_mag'][:, h + 1:h + 1 + T_pred]
        is_colliding = (contact_mag > 0.1).float()
        if is_colliding.sum() > 0:
            L_contact = (F.relu(v_toward_wall + 0.05) * is_colliding.squeeze(-1)).mean()
        else:
            L_contact = torch.tensor(0.0, device=L_dyn.device)

        # ── 课程调度 ──
        p = float(self._progress)
        # 物理约束在早期更强, 晚期衰减让数据驱动学习
        phys_scale = max(0.05, 1.0 - p * 0.9)
        # 能量约束始终较强
        energy_scale = 1.0
        # 边界/碰撞从第2 epoch 开始
        boundary_scale = min(1.0, max(0.0, (p - 0.05) * 4))
        contact_scale = boundary_scale
        # 死区/偏航约束从早期就生效
        yaw_scale = max(0.5, 1.0)  # 始终有效
        deadzone_scale = min(1.0, max(0.1, p * 2))

        L_total = (L_dyn
                   + self.w_rollout * rollout_alpha * L_rollout
                   + self.w_phys * phys_scale * L_phys
                   + self.w_energy * energy_scale * L_energy
                   + self.w_yaw_damp * yaw_scale * L_yaw_damp
                   + self.w_deadzone * deadzone_scale * L_deadzone
                   + self.w_residual * L_residual
                   + self.w_boundary * boundary_scale * L_boundary
                   + self.w_contact * contact_scale * L_contact)
        L_total = torch.nan_to_num(L_total, nan=0.0, posinf=10.0, neginf=0.0)

        metrics = {
            'loss_total': L_total.item(),
            'L_dyn': L_dyn.item(),
            'L_linvel': L_linvel.item(),
            'L_angvel': L_angvel.item(),
            'L_rollout': L_rollout.item(),
            'L_rt5': L_rt5.item(),
            'L_rt10': L_rt10.item(),
            'L_rt30': L_rt30.item(),
            'L_phys': L_phys.item(),
            'L_energy': L_energy.item(),
            'L_yaw_damp': L_yaw_damp.item(),
            'L_deadzone': L_deadzone.item(),
            'L_residual': L_residual.item(),
            'L_boundary': L_boundary.item(),
            'L_contact': L_contact.item(),
        }
        return L_total, metrics


# ═══════════════════════════════════════════════════════════════════
# 边界感知头设计 (供 v20 Dreamer 集成参考)
# ═══════════════════════════════════════════════════════════════════

"""
Phase 1: 训练 VelocityDynamics (本文件 + train_v2.py)
  输入: GT wall_dist (从数据集位置解析)
  输出: next velocity + acceleration

Phase 2: 在 v20 Dreamer 中集成

  方案 A (推荐): 边界感知头 — 与动力学引擎解耦，独立训练
  ─────────────────────────────────────────────────────────

  2.1 在 v20 WorldModel 中新增 BoundaryHead:
      ```
      class BoundaryHead(nn.Module):
          def __init__(self, deter_size=512, out_dim=6):
              self.net = nn.Sequential(
                  nn.Linear(deter_size, 256), nn.SiLU(),
                  nn.Linear(256, 128), nn.SiLU(),
                  nn.Linear(128, out_dim),     # 6-dim wall_dist
              )

          def forward(self, deter):
              return self.net(deter)
      ```

  2.2 训练时加入 boundary_loss:
      ```
      pred_wall_dist = self.boundary_head(post['deter'])
      gt_wall_dist = data['wall_dist']          # 从数据集预计算
      boundary_loss = Huber(pred_wall_dist, gt_wall_dist)
      ```

  2.3 推理时:
      ```
      deter = world_model.dynamics.img_step(state, action)
      wall_dist_pred = boundary_head(deter['deter'])
      vel_pred = velocity_dynamics(lin_vel, ang_vel, quat_pred,
                                    thruster, depth, wall_dist_pred, ...)
      ```

  方案 B (备选): 隐式边界编码 — 边界信息通过 latent space 传递
  ──────────────────────────────────────────────────────────────
  在 RSSM encoder 中把 wall_dist 作为额外的 proprio 输入:
      ```
      embed = encoder(image, concat[proprio, wall_dist_from_data])
      ```
  动力学引擎的 wall_dist 输入由 latent state 通过小小的"解码器"产生:
      ```
      wall_dist = wall_dist_decoder(z_t, h_t)
      ```
"""


# ═══════════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════════

import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def quat_rotate(q, v):
    """
    四元数旋转: body_frame → world_frame
    q: (B, 4) in (x, y, z, w) order
    v: (B, 3) body-frame vector
    Returns: (B, 3) world-frame vector
    """
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    u = torch.stack([qx, qy, qz], dim=-1)
    uv = torch.cross(u, v, dim=-1)
    uuv = torch.cross(u, uv, dim=-1)
    return v + 2.0 * (qw.unsqueeze(-1) * uv + uuv)


if __name__ == "__main__":
    model = VelocityDynamics({'jump_frame': 2, 'thruster_dim': 24})
    print(f"VelocityDynamics 参数量: {count_parameters(model):,}")
    print(f"  PhysicsParamNet: {count_parameters(model.physics_net):,}")
    print(f"  ResidualGRU:     {count_parameters(model.residual_gru):,}")

    B, T, h = 4, 42, 10
    batch = {
        'lin_vel': torch.randn(B, T, 3) * 0.3,
        'ang_vel': torch.randn(B, T, 3) * 0.1,
        'lin_accel': torch.randn(B, T, 3) * 0.5,
        'ang_accel': torch.randn(B, T, 3) * 0.1,
        'orientation': F.normalize(torch.randn(B, T, 4), dim=-1),
        'thruster': torch.randn(B, T, 24) * 0.5,
        'depth': torch.randn(B, T, 1) * 3,
        'altitude': torch.randn(B, T, 1) * 2,
        'wall_dist': torch.rand(B, T, 6) * 5,
        'wall_normals': torch.randn(B, T, 6, 3),
        'contact_mag': torch.rand(B, T, 1) * 0.1,
        'payload': torch.zeros(B, 1),
    }

    out = model(batch)
    print(f"\n前向传播成功!")
    print(f"  T_pred: {out['T_pred']}")
    print(f"  next_nu shape: {out['predictions']['next_nu'].shape}")
    print(f"  nu_dot shape:  {out['predictions']['nu_dot'].shape}")

    loss_fn = V2Loss()
    out['_h'] = h
    loss, metrics = loss_fn(out, batch)
    print(f"  Loss: {loss.item():.4f}, dyn={metrics['L_dyn']:.4f}, phys={metrics['L_phys']:.4f}")
    print("ALL TESTS PASSED")
