#!/usr/bin/env python3
"""
v2 VelocityDynamics 模拟器
加载训练好的 ckpt_v2_best.pt，给定初始状态和控制序列，推演 ROV 速度与轨迹

与 v1 区别:
  - 引擎只输出速度，不输出位置
  - 位置由速度积分得到（仅用于可视化）
  - wall_dist 由积分后的位置解析计算

用法:
  conda run -n ROS_env python dynamics_simulator_v2.py --ckpt ckpt_v2_best.pt --interactive
  conda run -n ROS_env python dynamics_simulator_v2.py --ckpt ckpt_v2_best.pt --pattern circle --steps 200 --output traj_v2.png
"""

import argparse
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
if '--interactive' in sys.argv:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        matplotlib.use('Agg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button, Slider

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

_V21_DIR = Path(__file__).resolve().parent
if str(_V21_DIR) not in sys.path:
    sys.path.insert(0, str(_V21_DIR))

from model_v2 import VelocityDynamics
from pool_geometry import POOL_X_MIN, POOL_X_MAX, POOL_Y_MIN, POOL_Y_MAX
from pool_geometry import pool_z_top, pool_z_bottom, compute_wall_distances

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _quat_from_yaw(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=np.float32)


def _yaw_from_quat(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _clamp_to_pool(x, y, z):
    x = max(POOL_X_MIN + 0.3, min(POOL_X_MAX - 0.3, x))
    y = max(POOL_Y_MIN + 0.3, min(POOL_Y_MAX - 0.3, y))
    z_top = pool_z_top(y)
    z_bot = pool_z_bottom(x, y)
    z = max(z_bot + 0.1, min(z_top - 0.1, z))
    return x, y, z


def _build_approx_norm_stats_v2(input_dim):
    """保守的近似归一化 (仅用于无法从 checkpoint 或数据集获取真实统计量时)"""
    mean = np.zeros(input_dim, dtype=np.float32)
    std = np.ones(input_dim, dtype=np.float32)

    mean[0:3] = [0.0, 0.0, 0.0];     std[0:3] = [0.2, 0.2, 0.15]     # lin_vel
    mean[3:6] = [0.0, 0.0, 0.0];     std[3:6] = [0.1, 0.1, 0.15]      # ang_vel
    mean[6:10] = [0.0, 0.0, 0.0, 0.0]; std[6:10] = [1., 1., 1., 1.]   # quat

    thr_start, thr_end = 10, 10 + input_dim - 10 - 9 - 3 - 1
    mean[thr_start:thr_end] = 0.0;    std[thr_start:thr_end] = 120.0
    env_start = thr_end
    mean[env_start + 0] = -5.0;       std[env_start + 0] = 3.0         # depth
    mean[env_start + 1] = 2.0;        std[env_start + 1] = 2.0         # altitude
    mean[env_start + 2:env_start + 8] = 3.0; std[env_start + 2:env_start + 8] = 2.0  # wall_dist
    mean[env_start + 8] = 0.0;        std[env_start + 8] = 0.5         # contact_mag
    acc_start = env_start + 9
    mean[acc_start:acc_start + 3] = 0.0; std[acc_start:acc_start + 3] = 0.5  # accel_proxy
    mean[-1] = 0.0;                   std[-1] = 1.0                    # payload

    return mean, std


def _build_norm_stats_from_dataset(dataset_root, jump_frame):
    """
    从数据集计算 v2 输入布局的真实归一化统计量 (与 train_v2.py 完全一致)

    v2 input layout: lin_vel(3) + ang_vel(3) + quat(4) + thr + depth(1)
                     + altitude(1) + wall_dist(6) + contact_mag(1)
                     + lin_accel(3) + payload(1)
    """
    from track_data_loader import build_dataloaders

    dl_train, _, _, _, ds_train, _, _ = build_dataloaders(
        dataset_root,
        window_steps=32, stride=8, target_hz=15.0,
        batch_size=4, num_workers=0, history_len=10,
        jump_frame=jump_frame,
    )
    ns = ds_train.norm

    dt_mult = jump_frame + 1
    thr_mean = np.tile(ns['thruster']['mean'], dt_mult)
    thr_std = np.tile(ns['thruster']['std'], dt_mult)

    v2_mean = np.concatenate([
        ns['lin_vel']['mean'],      # 3
        ns['ang_vel']['mean'],      # 3
        np.zeros(4, dtype=np.float32),  # quat
        thr_mean,                   # thruster
        ns['depth']['mean'],        # 1
        ns['altitude']['mean'],     # 1
        ns['wall_dist']['mean'],    # 6
        ns['contact_mag']['mean'],  # 1
        ns['lin_accel']['mean'],    # 3
        np.array([0.0], dtype=np.float32),  # payload
    ]).astype(np.float32)

    v2_std = np.concatenate([
        ns['lin_vel']['std'],
        ns['ang_vel']['std'],
        np.ones(4, dtype=np.float32),
        thr_std,
        ns['depth']['std'],
        ns['altitude']['std'],
        ns['wall_dist']['std'],
        ns['contact_mag']['std'],
        ns['lin_accel']['std'],
        np.array([1.0], dtype=np.float32),
    ]).astype(np.float32)

    return v2_mean, v2_std


class V2DynamicsSimulator:
    def __init__(self, checkpoint_path, dataset_root=None):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        meta = ckpt.get("meta", {})
        model_cfg = meta.get("model_config", {})

        model_cfg.setdefault('thruster_dim', 8)
        model_cfg.setdefault('env_dim', 9)
        model_cfg.setdefault('payload_dim', 1)
        model_cfg.setdefault('hist_len', 10)
        model_cfg.setdefault('physics_hidden', 256)
        model_cfg.setdefault('residual_hidden', 128)
        model_cfg.setdefault('residual_layers', 2)
        model_cfg.setdefault('cond_dim', 8)
        model_cfg.setdefault('jump_frame', 0)
        model_cfg.setdefault('target_hz', 15.0)

        # 兼容旧 checkpoint: 若无 m_rb_diag, 用旧值
        if 'm_rb_diag' not in model_cfg:
            old_m_rb = [660.0, 751.0, 1253.0, 179.0, 320.0, 236.0]
            print(f"  ⚠️ 旧 checkpoint (无 m_rb_diag), 使用旧 M_RB={old_m_rb}")
            model_cfg['m_rb_diag'] = old_m_rb

        self.jump_frame = model_cfg['jump_frame']
        self.dt = (self.jump_frame + 1) / model_cfg['target_hz']
        self.hist_len = model_cfg['hist_len']
        self.thruster_dim = model_cfg['thruster_dim']
        self.target_hz = model_cfg['target_hz']

        self.model = VelocityDynamics(model_cfg).to(DEVICE)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # ── 归一化统计量: 三级 fallback ──
        # 1. checkpoint meta 中保存的真实统计量 (最佳)
        # 2. 从数据集实时计算 (与 train_v2.py 完全一致)
        # 3. 硬编码近似 (仅作最后兜底)
        norm_mean, norm_std = None, None
        norm_source = None

        if 'norm_mean' in meta and 'norm_std' in meta:
            _mean = np.array(meta['norm_mean'], dtype=np.float32)
            _std = np.array(meta['norm_std'], dtype=np.float32)
            if len(_mean) == self.model.input_dim:
                norm_mean, norm_std = _mean, _std
                norm_source = "checkpoint"
            else:
                print(f"  ⚠️ checkpoint norm dim ({len(_mean)}) != model input_dim"
                      f" ({self.model.input_dim}), 跳过")

        if norm_mean is None and dataset_root is not None:
            try:
                print(f"  📂 从数据集计算归一化统计量: {dataset_root}")
                norm_mean, norm_std = _build_norm_stats_from_dataset(
                    dataset_root, self.jump_frame)
                if len(norm_mean) != self.model.input_dim:
                    print(f"  ⚠️ dataset norm dim ({len(norm_mean)}) != model input_dim"
                          f" ({self.model.input_dim}), 跳过")
                    norm_mean, norm_std = None, None
                else:
                    norm_source = "dataset"
            except Exception as e:
                print(f"  ⚠️ 数据集加载失败: {e}")

        if norm_mean is None:
            print(f"  ⚠️ 无法获取真实归一化统计量, 使用硬编码近似 (结果可能不准!)")
            norm_mean, norm_std = _build_approx_norm_stats_v2(self.model.input_dim)
            norm_source = "approx"

        self.model.set_norm_stats(
            torch.from_numpy(norm_mean).to(DEVICE),
            torch.from_numpy(norm_std).to(DEVICE),
        )

        # ── 环境特征默认值 (从训练分布提取, 防止 OOD 输入) ──
        # v2 layout: lin_vel(3) + ang_vel(3) + quat(4) + thr + env(9) + accel(3) + payload(1)
        _thr_end = 10 + self.model.input_dim - 10 - 9 - 3 - 1
        _env_s = _thr_end  # env 起始索引
        self._env_mean_alt = float(norm_mean[_env_s + 1])       # altitude mean
        self._env_std_alt = float(norm_std[_env_s + 1])         # altitude std
        self._env_mean_cm = float(norm_mean[_env_s + 8])        # contact_mag mean
        self._env_mean_accel = norm_mean[_env_s + 9:_env_s + 12].copy()  # accel mean (含重力)

        thr_std_avg = np.mean(norm_std[10:10 + self.thruster_dim])
        print(f"Model: {Path(checkpoint_path).name}")
        print(f"  jump_frame={self.jump_frame}  dt={self.dt:.3f}s  hist={self.hist_len}")
        print(f"  thruster_dim={self.thruster_dim}  input_dim={self.model.input_dim}")
        print(f"  norm_source={norm_source}  (thr std≈{thr_std_avg:.1f})")
        print(f"  env_defaults: alt={self._env_mean_alt:.3f}±{self._env_std_alt:.4f}"
              f"  accel_z={self._env_mean_accel[2]:.2f}  cm={self._env_mean_cm:.1f}")
        print(f"  device: {DEVICE}")

    @torch.no_grad()
    def simulate(self, init_linvel, init_angvel, init_quat, init_position,
                 thruster_sequence, n_steps=None):
        if n_steps is None:
            n_steps = len(thruster_sequence)

        to_t = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(DEVICE)

        lv = to_t(init_linvel)
        av = to_t(init_angvel)
        qu = to_t(init_quat)
        pos = to_t(init_position)

        # ── 计算初始环境特征 (与训练分布一致) ──
        init_wd, _ = compute_wall_distances(pos)
        init_x, init_y = float(pos[0, 0]), float(pos[0, 1])
        init_alt_raw = float(pos[0, 2]) - pool_z_bottom(init_x, init_y)
        # altitude 训练分布极窄 (std≈0.0025), 限制在 mean±5σ 防止 OOD
        init_alt = float(np.clip(
            init_alt_raw,
            self._env_mean_alt - 5 * self._env_std_alt,
            self._env_mean_alt + 5 * self._env_std_alt,
        ))
        cm_default = torch.ones(1, 1, device=DEVICE) * self._env_mean_cm
        accel_default = torch.from_numpy(
            self._env_mean_accel.reshape(1, 3).astype(np.float32)).to(DEVICE)

        hist_frames = []
        for _ in range(self.hist_len):
            frame = self.model._build_input(
                lv, av, qu,
                torch.zeros(1, self.thruster_dim, device=DEVICE),
                pos[:, 2:3],
                torch.ones(1, 1, device=DEVICE) * init_alt,
                init_wd,
                cm_default,
                accel_default,
                torch.zeros(1, 1, device=DEVICE),
            )
            hist_frames.append(frame)
        hist_input = torch.stack(hist_frames, dim=1)

        positions = [pos.cpu().numpy()[0].copy()]
        quats = [qu.cpu().numpy()[0].copy()]
        lin_vels = [lv.cpu().numpy()[0].copy()]
        ang_vels = [av.cpu().numpy()[0].copy()]

        # 初始 accel 用训练均值 (含重力分量), 后续由模型 nu_dot 更新
        ap_t = accel_default.clone()

        for step in range(n_steps):
            thr_raw = np.asarray(thruster_sequence[step % len(thruster_sequence)], dtype=np.float32)
            if self.jump_frame > 0:
                thr_exp = np.tile(thr_raw, self.jump_frame + 1)
            else:
                thr_exp = thr_raw
            thr_t = torch.from_numpy(thr_exp).unsqueeze(0).to(DEVICE)

            wd_t, wn_t = compute_wall_distances(pos)
            dep = pos[:, 2:3]
            # 从位置计算 altitude 并 clamp 到训练分布
            cur_x, cur_y = float(pos[0, 0]), float(pos[0, 1])
            cur_alt_raw = float(pos[0, 2]) - pool_z_bottom(cur_x, cur_y)
            cur_alt = float(np.clip(
                cur_alt_raw,
                self._env_mean_alt - 5 * self._env_std_alt,
                self._env_mean_alt + 5 * self._env_std_alt,
            ))
            alt = torch.ones(1, 1, device=DEVICE) * cur_alt
            cm_t = cm_default
            pay_t = torch.zeros(1, 1, device=DEVICE)

            next_nu, nu_dot, _, _ = self.model.forward_step(
                lv, av, qu, thr_t, dep, alt, wd_t, wn_t, cm_t, ap_t, pay_t, hist_input)

            next_nu = next_nu.detach()
            next_lv = next_nu[:, :3]
            next_av = next_nu[:, 3:6]

            ap_t = nu_dot[:, :3].detach()

            dq0 = self.model.quat_derivative(qu, av)
            dq1 = self.model.quat_derivative(F.normalize(qu + dq0 * self.dt, dim=-1), next_av)
            qu_mid = F.normalize(qu + (dq0 + dq1) * (0.25 * self.dt), dim=-1)
            qu = F.normalize(qu + (dq0 + dq1) * (0.5 * self.dt), dim=-1)

            R = quat_to_rotation_matrix_torch(qu_mid)
            avg_lv = (lv + next_lv) / 2.0
            pos_dot = torch.bmm(R, avg_lv.unsqueeze(-1)).squeeze(-1)
            pos = pos + pos_dot * self.dt

            positions.append(pos.cpu().numpy()[0].copy())
            quats.append(qu.cpu().numpy()[0].copy())
            lin_vels.append(next_lv.cpu().numpy()[0].copy())
            ang_vels.append(next_av.cpu().numpy()[0].copy())

            lv = next_lv
            av = next_av

            new_frame = self.model._build_input(lv, av, qu, thr_t, dep, alt, wd_t, cm_t, ap_t, pay_t)
            hist_input = torch.cat([hist_input[:, 1:, :], new_frame.unsqueeze(1)], dim=1)

        return {
            'position': np.stack(positions),
            'quat': np.stack(quats),
            'lin_vel': np.stack(lin_vels),
            'ang_vel': np.stack(ang_vels),
        }


def quat_to_rotation_matrix_torch(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


class InteractiveSimulatorV2:
    BASE_THRUST = 50.0  # 默认基础推力 (Gazebo deadzone 后 ~50N 是合理中档)

    def __init__(self, sim: V2DynamicsSimulator, base_thrust=None):
        self.sim = sim
        if base_thrust is not None:
            self.BASE_THRUST = base_thrust

        # 默认起始位置: y=-18 处池底约 -9.5m, ROV 在底部上方 ~0.7m (与训练分布一致)
        _default_z = pool_z_bottom(0.0, -18.0) + sim._env_mean_alt
        self.rov_pos = np.array([0.0, -18.0, _default_z], dtype=np.float32)
        self.rov_yaw = 0.0
        self.rov_quat = _quat_from_yaw(self.rov_yaw)
        self.rov_linvel = np.zeros(3, dtype=np.float32)
        self.rov_angvel = np.zeros(3, dtype=np.float32)

        self.active_command = None
        self.step_count = 0

        # ── DOF 幅值 (滑块值, 默认 1.0 = 100% 基础推力) ──
        self.scale_surge = 1.0      # 前/后
        self.scale_sway = 1.0       # 左/右
        self.scale_heave = 1.0      # 升/降
        self.scale_yaw = 1.0        # 旋转

        self.traj_pos = deque(maxlen=500)
        self.traj_pos.append(self.rov_pos.copy())

        self._build_pool_vertices()
        self._setup_figure()

    def _build_pool_vertices(self):
        self.pool_bottom = [
            [POOL_X_MIN, POOL_Y_MIN, -9.5],
            [POOL_X_MAX, POOL_Y_MIN, -9.5],
            [POOL_X_MAX, POOL_Y_MAX, -9.5],
            [POOL_X_MIN, POOL_Y_MAX, -9.5],
        ]
        n_seg = 20
        self.pool_walls = []
        for i in range(n_seg):
            y0 = POOL_Y_MIN + (POOL_Y_MAX - POOL_Y_MIN) * i / n_seg
            y1 = POOL_Y_MIN + (POOL_Y_MAX - POOL_Y_MIN) * (i + 1) / n_seg
            zt0, zt1 = pool_z_top(y0), pool_z_top(y1)
            self.pool_walls.append([
                [POOL_X_MAX, y0, -9.5], [POOL_X_MAX, y1, -9.5],
                [POOL_X_MAX, y1, zt1], [POOL_X_MAX, y0, zt0],
            ])
            self.pool_walls.append([
                [POOL_X_MIN, y0, -9.5], [POOL_X_MIN, y1, -9.5],
                [POOL_X_MIN, y1, zt1], [POOL_X_MIN, y0, zt0],
            ])

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title('V2 VelocityDynamics Simulator')
        # 3D 视图: 左侧, 给右侧滑块留空间
        self.ax3d = self.fig.add_axes([0.02, 0.12, 0.68, 0.85], projection='3d')
        self._setup_control_buttons()
        self._setup_thrust_sliders()

    def _setup_control_buttons(self):
        btn_w, btn_h = 0.08, 0.028
        cx, cy = 0.76, 0.55
        col_gap = 0.088
        row_gap = 0.032

        btn_cfgs = [
            (cx, cy + row_gap, 'forward', 'W: Fwd'),
            (cx, cy, 'backward', 'S: Back'),
            (cx - col_gap, cy, 'strafe_left', 'A: Left'),
            (cx + col_gap, cy, 'strafe_right', 'D: Right'),
            (cx, cy + 2 * row_gap, 'ascend', 'Shift: Up'),
            (cx, cy - row_gap, 'descend', 'Ctrl: Down'),
            (cx - col_gap, cy + row_gap, 'turn_left', 'Q: Turn L'),
            (cx + col_gap, cy + row_gap, 'turn_right', 'E: Turn R'),
            (cx, cy - 2 * row_gap, 'stop', 'Stop'),
            (cx, cy - 3 * row_gap, 'reset', 'Reset'),
        ]

        self._cmd_buttons = {}
        for x, y, cmd, label in btn_cfgs:
            ax = self.fig.add_axes([x, y, btn_w, btn_h])
            btn = Button(ax, label)
            if cmd == 'stop':
                btn.on_clicked(lambda event: self._set_command(None))
            elif cmd == 'reset':
                btn.on_clicked(self._on_reset)
            else:
                btn.on_clicked(lambda event, c=cmd: self._set_command(c))
            self._cmd_buttons[cmd] = btn

    def _setup_thrust_sliders(self):
        """四个滑块: 前/后 左/右 升/降 旋转 各一个幅值"""
        slider_w, slider_h = 0.15, 0.022
        sx, sy_start = 0.76, 0.28
        gap = 0.038

        slider_cfgs = [
            ('surge',  'Fwd/Back',   0.0, 2.0, 1.0),
            ('sway',   'Left/Right', 0.0, 2.0, 1.0),
            ('heave',  'Up/Down',    0.0, 2.0, 1.0),
            ('yaw',    'Turn',       0.0, 2.0, 1.0),
        ]

        self._sliders = {}
        for i, (key, label, vmin, vmax, vinit) in enumerate(slider_cfgs):
            y = sy_start - i * gap
            # 标签
            self.fig.text(sx, y + 0.014, label, fontsize=8, color='#cccccc',
                          fontfamily='monospace')
            # 滑块
            ax = self.fig.add_axes([sx, y, slider_w, slider_h])
            sl = Slider(ax, '', vmin, vmax, valinit=vinit, valfmt='%.2f')
            sl.on_changed(lambda val, k=key: self._on_slider(k, val))
            self._sliders[key] = sl
            # 微调 ax 样式
            ax.patch.set_alpha(0.3)
            sl.poly.set_facecolor('#4a90d9')
            sl.valtext.set_fontsize(7)

    def _on_slider(self, key, val):
        if key == 'surge':
            self.scale_surge = val
        elif key == 'sway':
            self.scale_sway = val
        elif key == 'heave':
            self.scale_heave = val
        elif key == 'yaw':
            self.scale_yaw = val

    def _set_command(self, cmd):
        self.active_command = cmd

    def _on_reset(self, event=None):
        _default_z = pool_z_bottom(0.0, -18.0) + self.sim._env_mean_alt
        self.rov_pos = np.array([0.0, -18.0, _default_z], dtype=np.float32)
        self.rov_yaw = 0.0
        self.rov_quat = _quat_from_yaw(self.rov_yaw)
        self.rov_linvel = np.zeros(3, dtype=np.float32)
        self.rov_angvel = np.zeros(3, dtype=np.float32)
        self.active_command = None
        self.step_count = 0
        self.traj_pos.clear()
        self.traj_pos.append(self.rov_pos.copy())

    def _cmd_to_thruster(self):
        t = np.zeros(8, dtype=np.float32)
        cmd = self.active_command
        if cmd is None:
            return t

        fwd = self.BASE_THRUST * self.scale_surge
        swy = self.BASE_THRUST * self.scale_sway
        h_up = self.BASE_THRUST * self.scale_heave
        yw  = self.BASE_THRUST * self.scale_yaw
        h_down = -h_up

        if cmd == 'forward':       t[1] = fwd;  t[3] = fwd
        elif cmd == 'backward':    t[0] = fwd;  t[2] = fwd
        elif cmd == 'strafe_left': t[0] = swy;  t[1] = swy
        elif cmd == 'strafe_right':t[2] = swy;  t[3] = swy
        elif cmd == 'ascend':      t[4:8] = h_up
        elif cmd == 'descend':     t[4:8] = h_down
        elif cmd == 'turn_left':   t[0] = yw;   t[3] = yw
        elif cmd == 'turn_right':  t[1] = yw;   t[2] = yw
        return t

    def _update_rov(self):
        thr = self._cmd_to_thruster()
        traj = self.sim.simulate(
            self.rov_linvel, self.rov_angvel, self.rov_quat, self.rov_pos,
            thruster_sequence=thr.reshape(1, 8), n_steps=1,
        )

        self.rov_pos = traj['position'][-1].copy()
        self.rov_quat = traj['quat'][-1].copy()
        self.rov_linvel = traj['lin_vel'][-1].copy()
        self.rov_angvel = traj['ang_vel'][-1].copy()
        self.rov_yaw = _yaw_from_quat(self.rov_quat)

        self.rov_pos[0], self.rov_pos[1], self.rov_pos[2] = _clamp_to_pool(
            self.rov_pos[0], self.rov_pos[1], self.rov_pos[2])
        self.traj_pos.append(self.rov_pos.copy())

    def _draw_pool(self):
        ax = self.ax3d
        ax.cla()

        bottom_poly = Poly3DCollection([self.pool_bottom], alpha=0.15,
                                        facecolor='#3d2b1f', edgecolor='#5a4030', linewidth=0.5)
        ax.add_collection3d(bottom_poly)
        for wall in self.pool_walls:
            wp = Poly3DCollection([wall], alpha=0.08, facecolor='#4a6741',
                                   edgecolor='#5a7751', linewidth=0.3)
            ax.add_collection3d(wp)

        traj = np.array(list(self.traj_pos))
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color='#00ff88', linewidth=1.5, alpha=0.8, label='Trajectory')

        ax.scatter(*self.rov_pos, c='lime', s=100, marker='o', label='ROV', zorder=5)

        fwd_len = 1.5
        fwd = np.array([math.cos(self.rov_yaw), math.sin(self.rov_yaw), 0.0]) * fwd_len
        ax.quiver(self.rov_pos[0], self.rov_pos[1], self.rov_pos[2],
                  fwd[0], fwd[1], fwd[2], color='yellow', arrow_length_ratio=0.3, linewidth=2)

        ax.set_xlim(POOL_X_MIN - 2, POOL_X_MAX + 2)
        ax.set_ylim(POOL_Y_MIN - 2, POOL_Y_MAX + 2)
        ax.set_zlim(-11, 2)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_facecolor('#1a1a2e')
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False

        cmd_name = self.active_command or 'STOP'
        ax.set_title(
            f'V2 VelocityDynamics | CMD: {cmd_name} | Step: {self.step_count}\n'
            f'Pos: ({self.rov_pos[0]:.1f}, {self.rov_pos[1]:.1f}, {self.rov_pos[2]:.1f}) | '
            f'Vel: ({np.linalg.norm(self.rov_linvel):.2f} m/s) | '
            f'AngVel: ({np.linalg.norm(self.rov_angvel):.2f} rad/s) | '
            f'Scale: Sg={self.scale_surge:.1f} Sw={self.scale_sway:.1f} Hv={self.scale_heave:.1f} Yw={self.scale_yaw:.1f}',
            fontsize=9, color='white', pad=10)
        ax.legend(loc='upper left', fontsize=7)

    def run(self):
        def update(frame):
            self._update_rov()
            self.step_count += 1
            self._draw_pool()
            return []
        self.anim = matplotlib.animation.FuncAnimation(
            self.fig, update, interval=80, blit=False, cache_frame_data=False)
        plt.show()


def _generate_test_sequence(n_steps, pattern='hover'):
    if pattern == 'hover':
        return np.zeros((n_steps, 8), dtype=np.float32)
    elif pattern == 'forward':
        seq = np.zeros((n_steps, 8), dtype=np.float32)
        seq[:, 1] = 50.0; seq[:, 3] = 50.0
        return seq
    elif pattern == 'circle':
        seq = np.zeros((n_steps, 8), dtype=np.float32)
        for i in range(n_steps):
            phase = 2 * math.pi * i / n_steps
            seq[i, 1] = 50.0 * math.cos(phase)
            seq[i, 3] = 50.0 * math.cos(phase)
            seq[i, 0] = 40.0 * math.sin(phase)
            seq[i, 2] = 40.0 * math.sin(phase)
        return seq
    elif pattern == 'ascend':
        seq = np.zeros((n_steps, 8), dtype=np.float32)
        seq[:, 4:8] = 60.0
        return seq
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def _make_v2_simulate_fn(sim):
    """v2 仿真适配器 → replay GUI 需要的回调"""
    def simulate_fn(init_state, thr_seq, n_steps):
        return sim.simulate(
            init_state['lin_vel'], init_state['ang_vel'],
            init_state['quat'], init_state['pos'],
            thr_seq, n_steps,
        )
    return simulate_fn


def main():
    parser = argparse.ArgumentParser(description="V2 VelocityDynamics Simulator")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--replay", action="store_true", help="轨迹回放对比模式")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="数据集根目录 (用于计算真实归一化统计量, 默认 ../dataset/)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--pattern", type=str, default="circle",
                        choices=["hover", "forward", "circle", "ascend"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--init-x", type=float, default=0.0)
    parser.add_argument("--init-y", type=float, default=-18.0)
    parser.add_argument("--init-z", type=float, default=None,
                        help="初始 Z 位置 (默认: 池底+训练均值高度)")
    parser.add_argument("--init-yaw", type=float, default=0.0)
    parser.add_argument("--init-lvx", type=float, default=0.0)
    parser.add_argument("--init-lvy", type=float, default=0.0)
    parser.add_argument("--init-lvz", type=float, default=0.0)
    parser.add_argument("--thrust-scale", type=float, default=50.0,
                        help="基础推力幅值 N (默认 50, 范围 10-200)")
    args = parser.parse_args()

    print("=" * 60)
    print("  V2 VelocityDynamics Simulator")
    print("=" * 60)

    # 数据集路径: 优先用命令行参数, 否则尝试默认路径
    dataset_root = args.dataset_root
    if dataset_root is None:
        default_ds = Path(__file__).resolve().parents[2] / "dataset"
        if default_ds.exists():
            dataset_root = str(default_ds)

    sim = V2DynamicsSimulator(args.ckpt, dataset_root=dataset_root)

    if args.replay:
        dataset_root = args.dataset_root or str(Path(__file__).resolve().parents[2] / "dataset")
        from trajectory_replay_base import run_replay_mode
        run_replay_mode(dataset_root, _make_v2_simulate_fn(sim), "V2 VelocityDynamics")
        return

    if args.interactive:
        print(f"\nKeyboard: W/S=fwd/back A/D=left/right Shift/Ctrl=up/down Q/E=turn Stop/Reset")
        print(f"Base thrust: {args.thrust_scale:.0f}N | Sliders: 0.0-2.0x")
        inter = InteractiveSimulatorV2(sim, base_thrust=args.thrust_scale)
        inter.run()
        return

    init_z = args.init_z if args.init_z is not None else (
        pool_z_bottom(args.init_x, args.init_y) + sim._env_mean_alt)
    init_pos = np.array([args.init_x, args.init_y, init_z], dtype=np.float32)
    init_quat = _quat_from_yaw(args.init_yaw)
    init_lv = np.array([args.init_lvx, args.init_lvy, args.init_lvz], dtype=np.float32)
    init_av = np.zeros(3, dtype=np.float32)

    thr_seq = _generate_test_sequence(args.steps, args.pattern)

    print(f"\nPattern={args.pattern} steps={args.steps}")
    print(f"  pos0=({init_pos[0]:.1f},{init_pos[1]:.1f},{init_pos[2]:.1f}) lv0=({init_lv[0]:.2f},{init_lv[1]:.2f},{init_lv[2]:.2f})")

    t0 = time.time()
    traj = sim.simulate(init_lv, init_av, init_quat, init_pos,
                        thruster_sequence=thr_seq, n_steps=args.steps)
    elapsed = time.time() - t0

    pos = traj['position']
    vel = traj['lin_vel']
    av = traj['ang_vel']

    print(f"\nResults ({elapsed:.2f}s):")
    print(f"  final pos: ({pos[-1,0]:.2f}, {pos[-1,1]:.2f}, {pos[-1,2]:.2f})")
    print(f"  displacement: {np.linalg.norm(pos[-1] - pos[0]):.2f} m")
    print(f"  max lin_vel: {np.max(np.linalg.norm(vel, axis=-1)):.3f} m/s")
    print(f"  max ang_vel: {np.max(np.linalg.norm(av, axis=-1)):.3f} rad/s")
    print(f"  no NaN: {not np.any(np.isnan(pos))}")

    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='#00ff88', linewidth=1.5)
    ax1.scatter(*pos[0], c='cyan', s=80, marker='o', label='Start')
    ax1.scatter(*pos[-1], c='red', s=80, marker='^', label='End')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title(f'Trajectory ({args.pattern})')
    ax1.legend(); ax1.set_facecolor('#1a1a2e')
    ax1.xaxis.pane.fill = False; ax1.yaxis.pane.fill = False; ax1.zaxis.pane.fill = False

    t_axis = np.arange(len(vel)) * sim.dt
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(t_axis, np.linalg.norm(vel, axis=-1), color='#ff6644', linewidth=1.5)
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('|v| (m/s)')
    ax2.set_title('Linear Velocity'); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(t_axis, np.rad2deg(np.linalg.norm(av, axis=-1)), color='#44aaff', linewidth=1.5)
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('|ω| (deg/s)')
    ax3.set_title('Angular Velocity'); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved: {args.output}")
    plt.show()


if __name__ == '__main__':
    main()
