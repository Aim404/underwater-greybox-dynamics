#!/usr/bin/env python3
"""
轨迹回放对比模块 — 仿真器 vs 真实轨迹

提供:
  - scan_dataset()     扫描数据集目录
  - show_selection_gui() 轨迹选择 GUI
  - load_trajectory_for_replay() 加载并重采样 H5 数据
  - TrajectoryReplayGUI 主回放 GUI (3D轨迹 + 进度条 + 预测按钮 + 速度对比侧边栏)

被 dynamics_simulator.py 和 dynamics_simulator_v2.py 共用
"""

import os
import glob
import math
import json
import time
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from track_data_loader import (
    POOL_X_MIN, POOL_X_MAX, POOL_Y_MIN, POOL_Y_MAX,
    pool_z_top, pool_z_bottom, compute_wall_distances,
    load_recording_data, resample_to_hz,
)

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

TARGET_HZ = 15.0

# ═══════════════════════════════════════════════════════════════════
# 数据集扫描
# ═══════════════════════════════════════════════════════════════════

def scan_dataset(dataset_root):
    """扫描数据集目录, 返回 {category: [{name, path}]}"""
    category_entries = {}
    if not os.path.isdir(dataset_root):
        return category_entries

    for item in sorted(os.listdir(dataset_root)):
        item_path = os.path.join(dataset_root, item)
        if not os.path.isdir(item_path) or item == 'standard_sample':
            continue
        recordings = []
        for sub in sorted(os.listdir(item_path)):
            sub_path = os.path.join(item_path, sub)
            if not os.path.isdir(sub_path) or not sub.startswith("recording_"):
                continue
            recordings.append({'name': sub, 'path': sub_path, 'category': item})
        if recordings:
            category_entries[item] = recordings
    return category_entries


def show_selection_gui(category_entries, dataset_root):
    """显示轨迹选择 GUI, 返回选中的 entry 或 None"""
    category_names = list(category_entries.keys())
    flat_items = []
    for cat in category_names:
        for entry in category_entries[cat]:
            flat_items.append(entry)

    if not flat_items:
        print("No trajectories found")
        return None

    ROW_HEIGHT = 0.35
    HEADER_HEIGHT = 0.5
    MARGIN_TOP = 0.3
    MARGIN_BOTTOM = 0.3
    VISIBLE_ROWS = 20
    fig_height = MARGIN_TOP + HEADER_HEIGHT + VISIBLE_ROWS * ROW_HEIGHT + MARGIN_BOTTOM
    fig_width = 12

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    fig.canvas.manager.set_window_title("Trajectory Replay — Select Recording")

    scroll_offset = [0]
    selected_entry = [None]

    def _draw_list():
        ax.clear()
        ax.set_xlim(0, fig_width)
        ax.set_ylim(0, fig_height)
        ax.axis('off')
        ax.set_facecolor('white')

        y_cursor = fig_height - MARGIN_TOP
        flat_idx = 0
        items_drawn = 0

        for cat in category_names:
            if y_cursor < MARGIN_BOTTOM + HEADER_HEIGHT:
                break
            ax.text(0.2, y_cursor, f"[{cat}]", ha='left', va='center',
                    fontsize=10, fontweight='bold', color='#1a1a2e')
            y_cursor -= HEADER_HEIGHT * 0.8

            for entry in category_entries[cat]:
                if flat_idx < scroll_offset[0]:
                    flat_idx += 1
                    continue
                if y_cursor < MARGIN_BOTTOM + ROW_HEIGHT / 2 or items_drawn >= VISIBLE_ROWS:
                    break

                row_y = y_cursor
                row_x0, row_w = 0.3, fig_width - 0.8
                row_h = ROW_HEIGHT * 0.85
                bg_color = '#f0f4f8' if items_drawn % 2 == 0 else '#ffffff'
                rect = plt.Rectangle((row_x0, row_y - row_h / 2), row_w, row_h,
                                     facecolor=bg_color, edgecolor='#ccddee', linewidth=0.8)
                ax.add_patch(rect)
                ax.text(row_x0 + 0.15, row_y, "▶", ha='left', va='center',
                        fontsize=9, color='#3b82f6', fontweight='bold')
                ax.text(row_x0 + 0.4, row_y, entry['name'], ha='left', va='center',
                        fontsize=9, color='#1e293b', fontfamily='monospace')
                flat_idx += 1
                items_drawn += 1
                y_cursor -= ROW_HEIGHT

            y_cursor -= ROW_HEIGHT * 0.3

        total_items = len(flat_items)
        max_scroll = max(0, total_items - VISIBLE_ROWS)
        if max_scroll > 0:
            bar_x, bar_w = fig_width - 0.25, 0.12
            track_h = fig_height - MARGIN_TOP - MARGIN_BOTTOM
            track_rect = plt.Rectangle((bar_x - bar_w / 2, MARGIN_BOTTOM), bar_w, track_h,
                                       facecolor='#e2e8f0', edgecolor='none')
            ax.add_patch(track_rect)
            thumb_h = max(0.8, track_h * (VISIBLE_ROWS / total_items))
            scroll_frac = scroll_offset[0] / max_scroll if max_scroll > 0 else 0
            thumb_y = MARGIN_BOTTOM + scroll_frac * (track_h - thumb_h)
            thumb = plt.Rectangle((bar_x - bar_w / 2, thumb_y), bar_w, thumb_h,
                                  facecolor='#64748b', edgecolor='#475569', linewidth=0.8)
            ax.add_patch(thumb)

        ax.set_title("Click a trajectory to replay  |  Scroll to browse",
                     fontsize=11, color='#475569', pad=8)
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes is not ax:
            return
        y_cursor = fig_height - MARGIN_TOP
        flat_idx = 0
        for cat in category_names:
            y_cursor -= HEADER_HEIGHT * 0.8
            for entry in category_entries[cat]:
                if flat_idx < scroll_offset[0]:
                    flat_idx += 1
                    y_cursor -= ROW_HEIGHT
                    continue
                row_h = ROW_HEIGHT * 0.85
                if abs(event.ydata - y_cursor) < row_h / 2 + 0.1:
                    selected_entry[0] = entry
                    print(f"\n>> Selected: [{entry['category']}] {entry['name']}")
                    plt.close(fig)
                    return
                flat_idx += 1
                y_cursor -= ROW_HEIGHT
            y_cursor -= ROW_HEIGHT * 0.3

    def _on_scroll(event):
        max_scroll = max(0, len(flat_items) - VISIBLE_ROWS)
        if event.button == 'up':
            scroll_offset[0] = max(0, scroll_offset[0] - 1)
        elif event.button == 'down':
            scroll_offset[0] = min(max_scroll, scroll_offset[0] + 1)
        _draw_list()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('scroll_event', _on_scroll)
    _draw_list()
    plt.show(block=True)
    return selected_entry[0]


# ═══════════════════════════════════════════════════════════════════
# 轨迹数据加载
# ═══════════════════════════════════════════════════════════════════

def load_trajectory_for_replay(recording_dir):
    """加载 recording 目录的 H5 数据, 重采样到 15Hz, 返回 dict"""
    raw = load_recording_data(recording_dir, verbose=True)
    if raw is None:
        return None

    # 提取关键字段
    ts_raw = raw.get('timestamp')
    if ts_raw is None:
        ts_raw = raw.get('pose_gt/dlrov/timestamp')
    if ts_raw is None:
        print("❌ 未找到 timestamp")
        return None

    ts_raw = np.asarray(ts_raw, dtype=np.float64).reshape(-1)
    T = len(ts_raw)
    if T < 10:
        print(f"❌ 轨迹太短: {T} 帧")
        return None

    # 构建统一数据
    dlrov_pos = raw.get('pose_gt/dlrov/position')
    dlrov_quat = raw.get('pose_gt/dlrov/orientation')
    uuv_pos = raw.get('pose_gt/uuv/position')
    uuv_quat = raw.get('pose_gt/uuv/orientation')
    dvl_vel = raw.get('dvl/velocity')
    dvl_alt = raw.get('dvl/altitude')
    depth = raw.get('pressure/depth')
    imu_angvel = raw.get('imu/angular_velocity')
    imu_accel = raw.get('imu/linear_acceleration')

    # 推力器输入
    thr_inputs = []
    for i in range(8):
        ti = raw.get(f'thruster/inputs/thruster{i}')
        if ti is not None:
            thr_inputs.append(np.asarray(ti).reshape(-1))
    if len(thr_inputs) == 8:
        thr_arr = np.stack(thr_inputs, axis=-1)  # (T, 8)
    else:
        thr_arr = np.zeros((T, 8), dtype=np.float32)

    # 截断到最小长度
    min_len = T
    for arr in [dlrov_pos, dlrov_quat, uuv_pos, uuv_quat, dvl_vel, depth, imu_angvel, imu_accel]:
        if arr is not None:
            min_len = min(min_len, len(arr))
    ts_raw = ts_raw[:min_len]
    T = min_len

    dlrov_pos = np.asarray(dlrov_pos[:T], dtype=np.float32) if dlrov_pos is not None else np.zeros((T, 3), dtype=np.float32)
    dlrov_quat = np.asarray(dlrov_quat[:T], dtype=np.float32) if dlrov_quat is not None else np.zeros((T, 4), dtype=np.float32)
    dlrov_quat[:, -1] = 1.0 if dlrov_quat is not None else 1.0
    uuv_pos = np.asarray(uuv_pos[:T], dtype=np.float32) if uuv_pos is not None else dlrov_pos.copy()
    uuv_quat = np.asarray(uuv_quat[:T], dtype=np.float32) if uuv_quat is not None else dlrov_quat.copy()
    dvl_vel = np.asarray(dvl_vel[:T], dtype=np.float32) if dvl_vel is not None else np.zeros((T, 3), dtype=np.float32)
    depth = np.asarray(depth[:T], dtype=np.float32).reshape(-1) if depth is not None else -dlrov_pos[:, 2]
    dvl_alt = np.asarray(dvl_alt[:T], dtype=np.float32).reshape(-1) if dvl_alt is not None else np.full(T, 2.0, dtype=np.float32)
    imu_angvel = np.asarray(imu_angvel[:T], dtype=np.float32) if imu_angvel is not None else np.zeros((T, 3), dtype=np.float32)
    imu_accel = np.asarray(imu_accel[:T], dtype=np.float32) if imu_accel is not None else np.zeros((T, 3), dtype=np.float32)
    thr_arr = np.asarray(thr_arr[:T], dtype=np.float32)

    # 重采样到 15Hz
    t_uniform = np.arange(ts_raw[0], ts_raw[-1], 1.0 / TARGET_HZ, dtype=np.float64)

    def _interp(val, t_uniform, ts_raw):
        if val.ndim == 1:
            return np.interp(t_uniform, ts_raw, val)
        else:
            out = np.zeros((len(t_uniform), val.shape[-1]), dtype=np.float32)
            for j in range(val.shape[-1]):
                out[:, j] = np.interp(t_uniform, ts_raw, val[:, j])
            return out

    out = {
        'timestamps': t_uniform,
        'dlrov_pos': _interp(dlrov_pos, t_uniform, ts_raw),
        'dlrov_quat': _interp(dlrov_quat, t_uniform, ts_raw),
        'uuv_pos': _interp(uuv_pos, t_uniform, ts_raw),
        'uuv_quat': _interp(uuv_quat, t_uniform, ts_raw),
        'dvl_vel': _interp(dvl_vel, t_uniform, ts_raw),
        'depth': _interp(depth, t_uniform, ts_raw),
        'dvl_alt': _interp(dvl_alt, t_uniform, ts_raw),
        'imu_angvel': _interp(imu_angvel, t_uniform, ts_raw),
        'imu_accel': _interp(imu_accel, t_uniform, ts_raw),
        'thruster': _interp(thr_arr, t_uniform, ts_raw),
        'num_frames': len(t_uniform),
    }

    # 归一化四元数
    q_norm = np.linalg.norm(out['dlrov_quat'], axis=-1, keepdims=True)
    out['dlrov_quat'] = out['dlrov_quat'] / np.maximum(q_norm, 1e-8)
    q_norm_uuv = np.linalg.norm(out['uuv_quat'], axis=-1, keepdims=True)
    out['uuv_quat'] = out['uuv_quat'] / np.maximum(q_norm_uuv, 1e-8)

    print(f"  ✅ 加载成功: {out['num_frames']} 帧 @ {TARGET_HZ}Hz "
          f"({out['num_frames'] / TARGET_HZ:.1f}s)")
    return out


# ═══════════════════════════════════════════════════════════════════
# 回放 GUI
# ═══════════════════════════════════════════════════════════════════

def _yaw_from_quat(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_from_yaw(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=np.float32)


class TrajectoryReplayGUI:
    """
    轨迹回放对比 GUI

    布局:
      ┌──────────────────────┬──────────────────┐
      │                      │  lin_vel_x       │
      │   3D Trajectory      │  lin_vel_y       │
      │   (DLROV + UUV)     │  lin_vel_z       │
      │   + 预测轨迹         │  ang_vel_x       │
      │                      │  ang_vel_y       │
      │                      │  ang_vel_z       │
      ├──────────────────────┴──────────────────┤
      │  [10s] [Rollout] [▶]  ═══slider═══     │
      └─────────────────────────────────────────┘

    simulate_fn 签名:
      simulate_fn(init_state, thr_seq, n_steps) -> dict {
          'position': (n_steps+1, 3),
          'lin_vel':  (n_steps+1, 3),
          'ang_vel':  (n_steps+1, 3),
          'quat':     (n_steps+1, 4),
      }
    """

    def __init__(self, traj_data, simulate_fn, simulator_label="PHONIC"):
        self.data = traj_data
        self.simulate_fn = simulate_fn
        self.sim_label = simulator_label

        self.num_frames = traj_data['num_frames']
        self.dt = 1.0 / TARGET_HZ
        self.total_time = self.num_frames * self.dt

        self.current_idx = 0
        self.is_playing = False
        self.play_timer = None

        # 预测缓存
        self.pred_10s = None     # (10s_frames+1, N)  紫色虚线
        self.pred_10s_start = 0
        self.pred_rollout = None # 从0开始的rollout
        self.pred_rollout_valid = False

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
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.canvas.manager.set_window_title(f'{self.sim_label} — Trajectory Replay')

        # 3D 主图
        self.ax3d = self.fig.add_axes([0.03, 0.12, 0.62, 0.85], projection='3d')

        # 速度对比子图 (6个) — 3线速度 + 3角速度
        vel_labels = [
            ('Vx (m/s)',      'Surge'),      # lin_vel_x
            ('Vy (m/s)',      'Sway'),       # lin_vel_y
            ('Vz (m/s)',      'Heave'),      # lin_vel_z
            (u'\u03c9x (rad/s)', 'Roll'),    # ang_vel_x
            (u'\u03c9y (rad/s)', 'Pitch'),   # ang_vel_y
            (u'\u03c9z (rad/s)', 'Yaw'),     # ang_vel_z
        ]
        self.vel_axes = []
        self.vel_gt_lines = []
        self.vel_pred_lines = []

        for i, (ylab, title) in enumerate(vel_labels):
            row, col = i // 3, i % 3
            ax_v = self.fig.add_axes([0.68 + col * 0.105, 0.12 + (1 - row) * 0.283, 0.10, 0.27])
            ax_v.set_title(title, fontsize=10, color='white', fontweight='bold')
            ax_v.set_ylabel(ylab, fontsize=6, color='#888888')
            ax_v.set_facecolor('#1a1a2e')
            ax_v.tick_params(colors='#888888', labelsize=5)
            ax_v.grid(True, alpha=0.2)
            gt_line, = ax_v.plot([], [], color='#00ff88', linewidth=0.8, label='GT')
            pred_line, = ax_v.plot([], [], color='#ff66cc', linewidth=0.8, linestyle='--', label='Pred')
            ax_v.axvline(x=0, color='#ffffff', alpha=0.4, linewidth=0.5)
            ax_v.legend(fontsize=5, loc='upper right')
            self.vel_axes.append(ax_v)
            self.vel_gt_lines.append(gt_line)
            self.vel_pred_lines.append(pred_line)

        # 速度图区域标题
        self.fig.text(0.82, 0.99, 'Velocity Comparison', ha='center',
                      fontsize=9, color='white', fontweight='bold')
        self.fig.text(0.82, 0.97, 'GT  —  Pred ···  Rollout ··', ha='center',
                      fontsize=6, color='#aaaaaa')

        # 进度条
        ax_slider = self.fig.add_axes([0.10, 0.04, 0.58, 0.025])
        self.slider = Slider(ax_slider, 'Time (s)', 0.0, self.total_time,
                             valinit=0.0, valstep=self.dt)

        # 播放按钮
        ax_play = self.fig.add_axes([0.69, 0.04, 0.03, 0.025])
        self.play_btn = Button(ax_play, '▶')
        self.play_btn.label.set_fontsize(10)

        # 按键1: 10s预测
        ax_btn1 = self.fig.add_axes([0.73, 0.04, 0.065, 0.025])
        self.btn_10s = Button(ax_btn1, '10s Pred')
        self.btn_10s.label.set_fontsize(8)

        # 按键2: Rollout
        ax_btn2 = self.fig.add_axes([0.80, 0.04, 0.065, 0.025])
        self.btn_rollout = Button(ax_btn2, 'Rollout 0→')
        self.btn_rollout.label.set_fontsize(8)

        # 信息文本
        self.info_text = self.fig.text(0.52, 0.985, "", ha='center', fontsize=9,
                                        color='white', fontweight='bold')

        self.slider.on_changed(self._on_slider_change)
        self.play_btn.on_clicked(self._on_play_click)
        self.btn_10s.on_clicked(self._on_10s_click)
        self.btn_rollout.on_clicked(self._on_rollout_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._update_display()

    # ── 预测生成 ──────────────────────────────────────────────

    def _generate_prediction(self, start_idx, n_steps):
        """从 start_idx 开始, 用 thruster 序列仿真 n_steps 步"""
        if n_steps <= 0 or start_idx + n_steps > self.num_frames:
            n_steps = self.num_frames - start_idx
        if n_steps <= 0:
            return None

        init_pos = self.data['dlrov_pos'][start_idx].copy()
        init_quat = self.data['dlrov_quat'][start_idx].copy()
        init_lv = self.data['dvl_vel'][start_idx].copy()
        init_av = self.data['imu_angvel'][start_idx].copy()
        thr_seq = self.data['thruster'][start_idx:start_idx + n_steps].copy()

        init_state = {
            'pos': init_pos, 'quat': init_quat,
            'lin_vel': init_lv, 'ang_vel': init_av,
        }

        try:
            result = self.simulate_fn(init_state, thr_seq, n_steps)
            return result
        except Exception as e:
            print(f"  ⚠️ 预测生成失败: {type(e).__name__}: {e}")
            return None

    def _on_10s_click(self, event):
        """按键1: 从当前位置预测10秒"""
        n_10s = min(int(10.0 / self.dt), self.num_frames - self.current_idx)
        if n_10s <= 1:
            print("⚠️ 轨迹太短, 无法生成10s预测")
            return
        print(f"⏳ 生成 {n_10s * self.dt:.1f}s 预测 (从 t={self.current_idx * self.dt:.1f}s)...")
        self.pred_10s = self._generate_prediction(self.current_idx, n_10s)
        self.pred_10s_start = self.current_idx
        self._update_display()

    def _on_rollout_click(self, event):
        """按键2: 从0时刻开始rollout"""
        n_all = self.num_frames - 1
        if n_all <= 1:
            return
        print(f"⏳ 生成完整 rollout ({n_all * self.dt:.1f}s)...")
        self.pred_rollout = self._generate_prediction(0, n_all)
        self.pred_rollout_valid = True
        self._update_display()

    # ── 播放控制 ──────────────────────────────────────────────

    def _on_play_click(self, event):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.label.set_text('||')
            if self.play_timer is None:
                self.play_timer = self.fig.canvas.new_timer(interval=int(1000 * self.dt))
                self.play_timer.add_callback(self._on_timer_tick)
            self.play_timer.start()
        else:
            self.play_btn.label.set_text('▶')
            if self.play_timer:
                self.play_timer.stop()

    def _on_timer_tick(self):
        if not self.is_playing:
            if self.play_timer:
                self.play_timer.stop()
            return
        if self.current_idx >= self.num_frames - 1:
            self.is_playing = False
            self.play_btn.label.set_text('▶')
            if self.play_timer:
                self.play_timer.stop()
            return
        self.current_idx = min(self.current_idx + 1, self.num_frames - 1)
        t = self.current_idx * self.dt
        self.slider.set_val(t)

    def _on_slider_change(self, val):
        self.current_idx = min(int(val / self.dt), self.num_frames - 1)
        self._update_display()

    def _on_key(self, event):
        if event.key == 'left':
            self.current_idx = max(0, self.current_idx - 1)
        elif event.key == 'right':
            self.current_idx = min(self.num_frames - 1, self.current_idx + 1)
        elif event.key == ' ':
            self._on_play_click(None)
            return
        elif event.key == '1':
            self._on_10s_click(None)
            return
        elif event.key == '2':
            self._on_rollout_click(None)
            return
        else:
            return
        t = self.current_idx * self.dt
        self.slider.set_val(t)

    # ── 绘制 ──────────────────────────────────────────────────

    def _update_display(self):
        self._draw_3d()
        self._draw_velocity_panels()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _draw_3d(self):
        ax = self.ax3d
        ax.cla()

        # 水池
        bottom_poly = Poly3DCollection([self.pool_bottom], alpha=0.12,
                                        facecolor='#3d2b1f', edgecolor='#5a4030', linewidth=0.3)
        ax.add_collection3d(bottom_poly)
        for wall in self.pool_walls:
            wp = Poly3DCollection([wall], alpha=0.06, facecolor='#4a6741',
                                   edgecolor='#5a7751', linewidth=0.2)
            ax.add_collection3d(wp)

        idx = self.current_idx

        # DLROV GT 轨迹 (绿色实线)
        dlrov = self.data['dlrov_pos']
        ax.plot(dlrov[:, 0], dlrov[:, 1], dlrov[:, 2],
                color='#00ff88', linewidth=1.0, alpha=0.5, label='DLROV GT')

        # UUV GT 轨迹 (青色实线)
        uuv = self.data['uuv_pos']
        if np.any(np.abs(uuv - dlrov) > 0.01):
            ax.plot(uuv[:, 0], uuv[:, 1], uuv[:, 2],
                    color='#00ccff', linewidth=0.8, alpha=0.4, label='UUV GT')

        # 当前位置标记
        ax.scatter(*dlrov[idx], c='lime', s=80, marker='o', zorder=10, label='Current')
        fwd_len = 1.5
        yaw = _yaw_from_quat(self.data['dlrov_quat'][idx])
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0]) * fwd_len
        ax.quiver(dlrov[idx, 0], dlrov[idx, 1], dlrov[idx, 2],
                  fwd[0], fwd[1], fwd[2], color='yellow', arrow_length_ratio=0.3, linewidth=1.5)

        # 10s 预测 (紫色虚线)
        if self.pred_10s is not None:
            ppos = self.pred_10s['position']
            if len(ppos) > 1:
                ax.plot(ppos[:, 0], ppos[:, 1], ppos[:, 2],
                        color='#ff66cc', linewidth=1.2, linestyle='--', alpha=0.8,
                        label=f'10s Pred ({len(ppos) * self.dt:.1f}s)')
                ax.scatter(*ppos[-1], c='#ff66cc', s=50, marker='x', zorder=9)

        # Rollout 预测 (橙色虚线)
        if self.pred_rollout is not None and self.pred_rollout_valid:
            rpos = self.pred_rollout['position']
            if len(rpos) > 1:
                ax.plot(rpos[:, 0], rpos[:, 1], rpos[:, 2],
                        color='#ffaa00', linewidth=0.8, linestyle=':', alpha=0.6,
                        label=f'Rollout ({len(rpos) * self.dt:.1f}s)')

        ax.set_xlim(POOL_X_MIN - 2, POOL_X_MAX + 2)
        ax.set_ylim(POOL_Y_MIN - 2, POOL_Y_MAX + 2)
        ax.set_zlim(-11, 2)
        ax.set_xlabel('X (m)', fontsize=7)
        ax.set_ylabel('Y (m)', fontsize=7)
        ax.set_zlabel('Z (m)', fontsize=7)
        ax.set_facecolor('#1a1a2e')
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.legend(loc='upper left', fontsize=6)
        ax.set_title(f'T={idx * self.dt:.1f}s / {self.total_time:.1f}s',
                     fontsize=9, color='white', pad=5)

    def _draw_velocity_panels(self):
        """绘制6个速度对比子图 (GT实线, 预测虚线)"""
        idx = self.current_idx
        WINDOW = 200  # 滑动窗口帧数
        half = WINDOW // 2

        start_f = max(0, idx - half)
        end_f = min(self.num_frames, idx + half)
        t_axis = np.arange(start_f, end_f) * self.dt

        # GT 数据
        gt_data = [
            self.data['dvl_vel'][:, 0],   # lin_vel_x
            self.data['dvl_vel'][:, 1],   # lin_vel_y
            self.data['dvl_vel'][:, 2],   # lin_vel_z
            self.data['imu_angvel'][:, 0], # ang_vel_x
            self.data['imu_angvel'][:, 1], # ang_vel_y
            self.data['imu_angvel'][:, 2], # ang_vel_z
        ]

        for i, (ax, gt_line, pred_line) in enumerate(
            zip(self.vel_axes, self.vel_gt_lines, self.vel_pred_lines)):
            ax.clear()
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='#888888', labelsize=5)
            ax.grid(True, alpha=0.2)

            # GT (全范围)
            gt_full = gt_data[i]
            if end_f - start_f > 0 and len(gt_full) > start_f:
                gt_win = gt_full[start_f:min(end_f, len(gt_full))]
                if len(gt_win) > 0:
                    ax.plot(t_axis[:len(gt_win)], gt_win, color='#00ff88', linewidth=0.7, alpha=0.7)

            # 当前竖线
            ax.axvline(x=idx * self.dt, color='#ffffff', alpha=0.4, linewidth=0.5)

            # 预测覆盖 (10s)
            if self.pred_10s is not None:
                p_idx = self.pred_10s_start
                if i < 3:
                    pred_vel = self.pred_10s['lin_vel']
                else:
                    pred_vel = self.pred_10s['ang_vel']
                j = i if i < 3 else i - 3
                n_pred = len(pred_vel)
                if n_pred > 0 and p_idx + n_pred <= self.num_frames:
                    pt = np.arange(p_idx, p_idx + n_pred) * self.dt
                    ax.plot(pt, pred_vel[:, j], color='#ff66cc', linewidth=0.7,
                            linestyle='--', alpha=0.6)

            # Rollout 预测
            if self.pred_rollout is not None and self.pred_rollout_valid:
                if i < 3:
                    pred_vel = self.pred_rollout['lin_vel']
                else:
                    pred_vel = self.pred_rollout['ang_vel']
                j = i if i < 3 else i - 3
                n_pred = len(pred_vel)
                if n_pred > 0:
                    pt = np.arange(n_pred) * self.dt
                    ax.plot(pt, pred_vel[:, j], color='#ffaa00', linewidth=0.5,
                            linestyle=':', alpha=0.4)

            ax.set_xlim(start_f * self.dt, (end_f - 1) * self.dt)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

    def _update_info(self):
        idx = self.current_idx
        dlrov = self.data['dlrov_pos'][idx]
        lv = self.data['dvl_vel'][idx]
        av = self.data['imu_angvel'][idx]
        thr = self.data['thruster'][idx]
        thr_norm = np.linalg.norm(thr)

        info_lines = [
            f"T={idx * self.dt:.1f}s | Frame {idx}/{self.num_frames}",
            f"Pos: ({dlrov[0]:.1f}, {dlrov[1]:.1f}, {dlrov[2]:.1f})",
            f"LinVel: |v|={np.linalg.norm(lv):.2f} m/s",
            f"AngVel: |ω|={np.linalg.norm(av):.2f} rad/s",
            f"Thrust: |τ|={thr_norm:.0f}",
        ]
        pred_info = []
        if self.pred_10s is not None:
            pred_info.append(f"10s Pred: t={self.pred_10s_start * self.dt:.1f}s")
        if self.pred_rollout is not None:
            pred_info.append("Rollout: ready")
        if pred_info:
            info_lines.append(" | ".join(pred_info))

        self.info_text.set_text("  |  ".join(info_lines))

    def run(self):
        plt.show()


def run_replay(recording_dir, simulate_fn, simulator_label="PHONIC"):
    """运行回放模式的便捷入口"""
    print(f"\n📂 加载轨迹: {recording_dir}")
    traj = load_trajectory_for_replay(recording_dir)
    if traj is None:
        print("❌ 无法加载轨迹数据")
        return

    print(f"\n🎬 启动回放 GUI ({simulator_label})")
    print("   ← → 键: 逐帧移动    空格: 播放/暂停")
    print("   1: 10s预测  2: Rollout")
    gui = TrajectoryReplayGUI(traj, simulate_fn, simulator_label)
    gui.run()


def run_replay_mode(dataset_root, simulate_fn, simulator_label="PHONIC"):
    """完整的 replay 模式入口: 扫描 → 选择 → 加载 → 回放"""
    print(f"🔍 扫描数据集: {dataset_root}")
    entries = scan_dataset(dataset_root)
    if not entries:
        print("❌ 未找到任何轨迹文件夹")
        return

    total = sum(len(v) for v in entries.values())
    print(f"   找到 {len(entries)} 个类别, 共 {total} 个轨迹")

    selected = show_selection_gui(entries, dataset_root)
    if selected is None:
        print("👋 未选择轨迹, 退出")
        return

    run_replay(selected['path'], simulate_fn, simulator_label)
