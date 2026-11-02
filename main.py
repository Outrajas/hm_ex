"""Unified K-G-R State, Surface & Spatial Simulator.

Consolidates:
- Mode 1 [F1]: Interactive K-G-R Studio (Sliders, 6-DoF POV Orbit, Whiskers, Clearance Stalks)
- Mode 2 [F2]: Smooth Motion Choreography & Live Video Recording ([V] key)
- Mode 3 [F3]: 3-Panel Temporal Prediction & Timeline Studio (GT vs Prior vs Error Overlay)
"""

import sys
from pathlib import Path
import time
import importlib
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.import_module("00_spec.state_definition")
_coord = importlib.import_module("00_spec.coordinate_system")
_topo = importlib.import_module("00_spec.hand_topology")
_state_engine = importlib.import_module("02_state.state")
_const = importlib.import_module("02_state.constraints")
_fk = importlib.import_module("02_state.forward_kinematics")
_mano = importlib.import_module("02_state.mano_surface")
_rel = importlib.import_module("03_observation.relationship")
_ds = importlib.import_module("06_training.dataset")

HandState = _spec.HandState
ControlInput = _state_engine.ControlInput
state_update = _state_engine.state_update
project_constraints = _const.project_constraints
forward_kinematics = _fk.forward_kinematics
ManoSurface = _mano.ManoSurface
compute_skeleton_geometry = _rel.compute_skeleton_geometry
compute_glove_geometry = _rel.compute_glove_geometry
evaluate_relationship = _rel.evaluate_relationship
evaluate_consistency = _rel.evaluate_consistency
euler_to_quaternion = _coord.euler_to_quaternion
PARENT_INDICES = _topo.PARENT_INDICES
BONES = _rel.BONES
MANO_TIPS = _rel.MANO_TIPS
build_split_dataset = _ds.build_split_dataset

# Canvas Dimensions
CANVAS_W, CANVAS_H = 1440, 820
PANEL_W = 340
VIEW_W = CANVAS_W - PANEL_W
FOCAL_PX = 720.0
Y_FLOOR = 0.22
PALM_ARCH = [(1, 5), (5, 9), (9, 13), (13, 17)]

# Active Mode: 1=Interactive Studio, 2=Smooth Motion Test, 3=3-Panel Prediction
active_app_mode = 1

# Camera POV Orbit Parameters
cam_orbit_yaw = 0.0
cam_orbit_pitch = 0.20
is_orbiting_cam = is_dragging_xy = is_rotating_hand = is_depth_roll = False
last_mouse_x = last_mouse_y = 0

# Interactive Controls State
SLIDER_SPECS = [
    ["pos_x",       "Pos X",         -350,  350,    0, True],
    ["pos_y",       "Pos Y",         -300,  200,  -30, True],
    ["pos_z",       "Pos Z (Depth)",  250, 1200,  520, True],
    ["pitch",       "Hand Pitch",       0,  360,  180, True],
    ["yaw",         "Hand Yaw",         0,  360,  180, True],
    ["roll",        "Hand Roll",        0,  360,  180, True],
    ["splay",       "Knuckle Splay",    0,   45,   15, True],
    ["t_flex",      "Thumb Flex",       0,  100,    0, True],
    ["t_abd",       "Thumb Abd",        0,  100,    0, True],
    ["i_curl",      "Index Curl",       0,  100,    0, True],
    ["m_curl",      "Middle Curl",      0,  100,    0, True],
    ["r_curl",      "Ring Curl",        0,  100,    0, True],
    ["p_curl",      "Pinky Curl",       0,  100,    0, True],
]
slider_vals = {s[0]: s[4] for s in SLIDER_SPECS}
active_slider_key = None
is_auto_motion = False
ui_view_mode = 4  # 1: Skeleton, 2: MANO Mesh, 3: MANO+Skel, 4: Full K-G-R
ui_show_whiskers = True
ui_show_clearance = True

# Video Recording State (Mode 2)
is_recording = False
video_writer = None
prev_mesh_verts = None
max_jitter_mm = 0.0

# 3-Panel State (Mode 3)
test_dataset_batch = None
pred_frame_idx = 0
pred_is_paused = False
pred_timeline_mode = False
pred_error_scale = 1.0

def get_slider_layout(idx):
    y_start = 145 + idx * 37
    bx0, bx1 = 150, 310
    return y_start, bx0, bx1

def project_viewport_3d(pts_3d: np.ndarray, R_cam: np.ndarray, T_cam: np.ndarray, offset_x: int = PANEL_W, view_w: int = VIEW_W):
    pts_cam = (pts_3d - T_cam) @ R_cam.T
    Z = pts_cam[:, 2]
    valid = Z > 0.04
    safe_z = np.where(valid, Z, 1.0)
    u = (FOCAL_PX * (pts_cam[:, 0] / safe_z)) + (view_w * 0.5) + offset_x
    v = (FOCAL_PX * (pts_cam[:, 1] / safe_z)) + (CANVAS_H * 0.5)
    return np.column_stack([u, v]), valid

def sample_manual_input() -> ControlInput:
    target_T = np.array([slider_vals["pos_x"] / 1000.0, slider_vals["pos_y"] / 1000.0, slider_vals["pos_z"] / 1000.0], dtype=np.float64)
    r_rad = np.radians(slider_vals["roll"] - 180)
    p_rad = np.radians(slider_vals["pitch"] - 180)
    y_rad = np.radians(slider_vals["yaw"] - 180)
    target_R = euler_to_quaternion(r_rad, p_rad, y_rad)

    target_q = np.zeros(21, dtype=np.float64)
    target_q[0] = (slider_vals["t_abd"] / 100.0) * 0.6
    target_q[1] = (slider_vals["t_flex"] / 100.0) * 0.5
    target_q[2] = (slider_vals["t_abd"] / 100.0) * 0.3
    target_q[3] = (slider_vals["t_flex"] / 100.0) * 0.8
    target_q[4] = (slider_vals["t_flex"] / 100.0) * 1.2

    splay_rad = np.radians(slider_vals["splay"])
    curls = [
        (slider_vals["i_curl"] / 100.0,  splay_rad * 1.0, 5),
        (slider_vals["m_curl"] / 100.0,  0.0,             9),
        (slider_vals["r_curl"] / 100.0, -splay_rad * 0.7, 13),
        (slider_vals["p_curl"] / 100.0, -splay_rad * 1.4, 17)
    ]
    for c_val, s_val, q_idx in curls:
        target_q[q_idx]     = s_val
        target_q[q_idx + 1] = c_val * 1.45
        target_q[q_idx + 2] = c_val * 1.65
        target_q[q_idx + 3] = c_val * 1.30

    return ControlInput(target_T=target_T, target_R=target_R, target_q=target_q, authority=1.0)

def evaluate_fast_motion(t: float) -> tuple[ControlInput, str]:
    cycle = t % 6.0
    BASE_YAW = np.pi
    pos_x, pos_y, pos_z = 0.0, 0.02, 0.52
    roll, pitch, yaw = 0.0, 0.0, 0.0
    target_q = np.zeros(21, dtype=np.float64)
    target_q[5]  = np.radians(8.0)
    target_q[13] = np.radians(-6.0)
    target_q[17] = np.radians(-14.0)

    if cycle < 1.5:
        phase = "FAST SNAP CLENCH (1.8 Hz)"
        a = float(np.clip(0.5 - 0.5 * np.cos((cycle / 1.5) * 2.0 * np.pi * 1.8), 0.0, 1.0))
        target_q[0], target_q[1], target_q[3], target_q[4] = a * 0.5, a * 0.6, a * 0.9, a * 1.3
        for idx in [5, 9, 13, 17]:
            target_q[idx+1], target_q[idx+2], target_q[idx+3] = a * 1.55, a * 1.70, a * 1.35
    elif cycle < 3.0:
        phase = "VIGOROUS WAVING (4.5 Hz)"
        sub_t = cycle - 1.5
        w_freq = 2.0 * np.pi * 4.5
        roll = np.radians(34.0 * np.sin(w_freq * sub_t))
        yaw  = np.radians(16.0 * np.cos(w_freq * sub_t))
        pos_x = 0.035 * np.sin(w_freq * sub_t)
        fl = 0.20 + 0.20 * np.sin(w_freq * sub_t)
        for idx in [5, 9, 13, 17]:
            target_q[idx+1], target_q[idx+2] = fl * 0.8, fl * 1.0
    elif cycle < 4.5:
        phase = "FINGER RIPPLES (6.0 Hz)"
        sub_t = cycle - 3.0
        r_freq = 2.0 * np.pi * 6.0
        for d_i, q_base in enumerate([5, 9, 13, 17]):
            c = float(0.5 + 0.5 * np.sin(r_freq * sub_t + d_i * 1.2))
            target_q[q_base+1], target_q[q_base+2], target_q[q_base+3] = c * 1.45, c * 1.65, c * 1.25
        pitch = np.radians(12.0 * np.sin(r_freq * sub_t * 0.5))
    else:
        phase = "3D SPATIAL TUMBLE"
        sub_t = cycle - 4.5
        d_freq = 2.0 * np.pi * 2.0
        pos_x = 0.10 * np.sin(d_freq * sub_t)
        pos_y = 0.04 + 0.05 * np.cos(d_freq * sub_t)
        pos_z = 0.52 + 0.12 * np.sin(d_freq * sub_t * 1.5)
        pitch = np.radians(30.0 * np.sin(d_freq * sub_t))
        roll  = np.radians(22.0 * np.cos(d_freq * sub_t))

    target_T = np.array([pos_x, pos_y, pos_z], dtype=np.float64)
    target_R = euler_to_quaternion(roll, pitch, BASE_YAW + yaw)
    return ControlInput(target_T=target_T, target_R=target_R, target_q=target_q, authority=1.0), phase

def evaluate_smooth_choreography(t: float) -> tuple[ControlInput, str]:
    cycle = t % 12.0
    BASE_YAW = np.pi
    pos_x, pos_y, pos_z = 0.0, 0.02, 0.52
    roll, pitch, yaw = 0.0, 0.0, 0.0
    target_q = np.zeros(21, dtype=np.float64)
    target_q[5]  = np.radians(8.0)
    target_q[13] = np.radians(-6.0)
    target_q[17] = np.radians(-14.0)

    def smoothstep(v):
        cl = float(np.clip(v, 0.0, 1.0))
        return cl * cl * (3.0 - 2.0 * cl)

    if cycle < 2.5:
        phase = "1. Facing Camera (Open Palm)"
        pos_y += 0.006 * np.sin(2.0 * np.pi * (cycle / 2.5))
        roll = np.radians(2.0 * np.sin(2.0 * np.pi * (cycle / 2.5)))
    elif cycle < 5.5:
        phase = "2. Smooth Fist Clench"
        alpha = smoothstep((cycle - 2.5) / 3.0)
        target_q[0], target_q[1], target_q[3], target_q[4] = alpha*0.45, alpha*0.55, alpha*0.85, alpha*1.25
        for idx in [5, 9, 13, 17]:
            target_q[idx+1], target_q[idx+2], target_q[idx+3] = alpha*1.50, alpha*1.65, alpha*1.30
    elif cycle < 8.0:
        phase = "3. Smooth Palm Release"
        alpha = 1.0 - smoothstep((cycle - 5.5) / 2.5)
        target_q[0], target_q[1], target_q[3], target_q[4] = alpha*0.45, alpha*0.55, alpha*0.85, alpha*1.25
        for idx in [5, 9, 13, 17]:
            target_q[idx+1], target_q[idx+2], target_q[idx+3] = alpha*1.50, alpha*1.65, alpha*1.30
    elif cycle < 10.5:
        phase = "4. Harmonic Waving"
        sub_t = cycle - 8.0
        w_freq = 1.6
        roll = np.radians(28.0 * np.sin(2.0 * np.pi * w_freq * sub_t))
        yaw  = np.radians(14.0 * np.cos(2.0 * np.pi * w_freq * sub_t))
        pos_x = 0.03 * np.sin(2.0 * np.pi * w_freq * sub_t)
        fl = 0.15 + 0.15 * np.sin(2.0 * np.pi * w_freq * sub_t)
        for idx in [5, 9, 13, 17]:
            target_q[idx+1], target_q[idx+2] = fl * 0.7, fl * 1.0
    else:
        phase = "5. 3D Spatial Depth Orbit"
        sub_t = (cycle - 10.5) / 1.5
        s_curve = np.sin(np.pi * sub_t)
        pos_z = 0.52 - 0.14 * s_curve
        pitch = np.radians(22.0 * s_curve)
        yaw   = np.radians(-18.0 * s_curve)

    target_T = np.array([pos_x, pos_y, pos_z], dtype=np.float64)
    target_R = euler_to_quaternion(roll, pitch, BASE_YAW + yaw)
    return ControlInput(target_T=target_T, target_R=target_R, target_q=target_q, authority=1.0), phase

def on_mouse(event, x, y, flags, userdata):
    global is_dragging_xy, is_rotating_hand, is_depth_roll, is_orbiting_cam
    global last_mouse_x, last_mouse_y, cam_orbit_yaw, cam_orbit_pitch
    global active_slider_key, is_auto_motion, ui_show_whiskers, ui_show_clearance, ui_view_mode
    global active_app_mode

    # Top Tab Navigation Bar (y < 42)
    if y < 42 and event == cv2.EVENT_LBUTTONDOWN:
        if 20 <= x < 320:   active_app_mode = 1
        elif 330 <= x < 630: active_app_mode = 2
        elif 640 <= x < 960: active_app_mode = 3
        return

    # Mode 1 Left Panel Interaction
    if active_app_mode == 1 and x < PANEL_W:
        if event == cv2.EVENT_LBUTTONDOWN:
            if 70 <= y <= 105 and 20 <= x <= 320:
                is_auto_motion = not is_auto_motion
                return
            if 655 <= y <= 685 and 20 <= x <= 160:
                ui_show_whiskers = not ui_show_whiskers
                return
            if 655 <= y <= 685 and 170 <= x <= 320:
                ui_show_clearance = not ui_show_clearance
                return
            if 710 <= y <= 745 and 20 <= x <= 320:
                col_w = 75
                b_idx = int((x - 20) // col_w) + 1
                if 1 <= b_idx <= 4:
                    ui_view_mode = b_idx
                return

            if not is_auto_motion:
                for idx, (k, lbl, mn, mx, val, is_i) in enumerate(SLIDER_SPECS):
                    y_s, bx0, bx1 = get_slider_layout(idx)
                    if (y_s - 12) <= y <= (y_s + 12) and (bx0 - 10) <= x <= (bx1 + 10):
                        active_slider_key = k
                        frac = float(np.clip((x - bx0) / float(bx1 - bx0), 0.0, 1.0))
                        slider_vals[k] = int(round(mn + frac * (mx - mn))) if is_i else float(mn + frac * (mx - mn))
                        return

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON) and active_slider_key:
            if not is_auto_motion:
                for idx, (k, lbl, mn, mx, val, is_i) in enumerate(SLIDER_SPECS):
                    if k == active_slider_key:
                        _, bx0, bx1 = get_slider_layout(idx)
                        frac = float(np.clip((x - bx0) / float(bx1 - bx0), 0.0, 1.0))
                        slider_vals[k] = int(round(mn + frac * (mx - mn))) if is_i else float(mn + frac * (mx - mn))
                        return

        elif event == cv2.EVENT_LBUTTONUP:
            active_slider_key = None
        return

    # Viewport 3D Navigation
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY: is_depth_roll = True
        elif flags & cv2.EVENT_FLAG_CTRLKEY: is_orbiting_cam = True
        else: is_dragging_xy = True
        last_mouse_x, last_mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging_xy = is_depth_roll = is_orbiting_cam = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        if flags & (cv2.EVENT_FLAG_SHIFTKEY | cv2.EVENT_FLAG_CTRLKEY): is_orbiting_cam = True
        else: is_rotating_hand = True
        last_mouse_x, last_mouse_y = x, y
    elif event == cv2.EVENT_RBUTTONUP:
        is_rotating_hand = is_orbiting_cam = False
    elif event == cv2.EVENT_MBUTTONDOWN:
        is_orbiting_cam = True
        last_mouse_x, last_mouse_y = x, y
    elif event == cv2.EVENT_MBUTTONUP:
        is_orbiting_cam = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        delta = 30 if flags > 0 else -30
        if active_app_mode == 1 and not is_auto_motion:
            slider_vals["pos_z"] = int(np.clip(slider_vals["pos_z"] - delta, 250, 1200))
    elif event == cv2.EVENT_MOUSEMOVE:
        dx, dy = x - last_mouse_x, y - last_mouse_y
        if is_orbiting_cam:
            cam_orbit_yaw -= dx * 0.008
            cam_orbit_pitch = float(np.clip(cam_orbit_pitch - dy * 0.008, -0.65, 0.85))
            last_mouse_x, last_mouse_y = x, y
        elif is_dragging_xy and active_app_mode == 1 and not is_auto_motion:
            slider_vals["pos_x"] = int(np.clip(slider_vals["pos_x"] + dx, -350, 350))
            slider_vals["pos_y"] = int(np.clip(slider_vals["pos_y"] + dy, -300, 200))
            last_mouse_x, last_mouse_y = x, y
        elif is_rotating_hand and active_app_mode == 1 and not is_auto_motion:
            slider_vals["yaw"] = (slider_vals["yaw"] + dx + 360) % 360
            slider_vals["pitch"] = (slider_vals["pitch"] - dy + 360) % 360
            last_mouse_x, last_mouse_y = x, y
        elif is_depth_roll and active_app_mode == 1 and not is_auto_motion:
            slider_vals["roll"] = (slider_vals["roll"] + dx + 360) % 360
            slider_vals["pos_z"] = int(np.clip(slider_vals["pos_z"] + dy * 2, 250, 1200))
            last_mouse_x, last_mouse_y = x, y

def render_top_navigation(canvas: np.ndarray):
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, 42), (18, 18, 22), -1)
    cv2.line(canvas, (0, 42), (CANVAS_W, 42), (55, 55, 68), 1)

    tabs = [
        (1, "[F1] Interactive K-G-R Studio", 20, 320),
        (2, "[F2] Smooth Motion & Video Test", 330, 630),
        (3, "[F3] 3-Panel Temporal Prediction", 640, 960)
    ]
    for m_id, label, x0, x1 in tabs:
        active = (active_app_mode == m_id)
        bg_col = (0, 140, 220) if active else (30, 30, 38)
        cv2.rectangle(canvas, (x0, 6), (x1, 36), bg_col, -1)
        cv2.rectangle(canvas, (x0, 6), (x1, 36), (80, 80, 100), 1)
        t_col = (255, 255, 255) if active else (180, 180, 195)
        cv2.putText(canvas, label, (x0 + 14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, t_col, 1)

def run_main():
    global active_app_mode, is_auto_motion, ui_view_mode, ui_show_whiskers, ui_show_clearance
    global is_recording, video_writer, prev_mesh_verts, max_jitter_mm
    global pred_frame_idx, pred_is_paused, pred_timeline_mode, pred_error_scale

    mano = ManoSurface()
    splits = build_split_dataset(n_trajectories=600, frames_per_traj=50, seed=101)
    test_split = splits["test"]

    cv2.namedWindow("K-G-R Unified State & Simulator Studio", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("K-G-R Unified State & Simulator Studio", on_mouse)

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    persistent_state = HandState.neutral(z_depth=0.52)

    grid_pts_3d = []
    for gx in np.linspace(-0.35, 0.35, 11):
        grid_pts_3d.extend([[gx, Y_FLOOR, 0.20], [gx, Y_FLOOR, 0.85]])
    for gz in np.linspace(0.20, 0.85, 10):
        grid_pts_3d.extend([[-0.35, Y_FLOOR, gz], [0.35, Y_FLOOR, gz]])
    grid_pts_3d = np.array(grid_pts_3d, dtype=np.float64)

    light_dir = np.array([0.4, -0.7, -0.6], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)

    start_clock = time.perf_counter()
    last_clock = start_clock
    fps_smooth = 120.0

    print("=================================================================")
    print(" K-G-R UNIFIED STATE, SURFACE & TEMPORAL SIMULATOR")
    print(" Tabs: [F1] Interactive Studio | [F2] Smooth Motion | [F3] 3-Panel")
    print(" Keys: [SPACE] Pause/Action | [V] Record MP4 | [M] Mode | [ESC] Exit")
    print("=================================================================")

    while True:
        t0 = time.perf_counter()
        sim_time = t0 - start_clock
        dt = max(t0 - last_clock, 1e-4)
        last_clock = t0

        canvas[:] = (22, 22, 28)

        # Camera POV transformation
        scene_center = np.array([0.0, -0.02, 0.52])
        cy, sy = np.cos(cam_orbit_yaw), np.sin(cam_orbit_yaw)
        cp, sp = np.cos(cam_orbit_pitch), np.sin(cam_orbit_pitch)
        R_cam = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        T_cam = scene_center - R_cam.T @ np.array([0.0, 0.0, 0.52])

        # -------------------------------------------------------------
        # TAB 1: INTERACTIVE K-G-R STUDIO
        # -------------------------------------------------------------
        if active_app_mode == 1:
            if is_auto_motion:
                ctrl_in, phase_label = evaluate_fast_motion(sim_time)
            else:
                ctrl_in = sample_manual_input()
                phase_label = "MANUAL CONTROL (Sliders + Mouse POV)"

            raw_state = state_update(persistent_state, ctrl_in, dt)
            persistent_state = project_constraints(raw_state)

            jp, jr = forward_kinematics(persistent_state)
            K = compute_skeleton_geometry(jp, jr)

            if mano.is_available and ui_view_mode in [2, 3, 4]:
                vm, fm, k21 = mano.forward(persistent_state, jp)
                G = compute_glove_geometry(vm, fm, k21)
                R = evaluate_relationship(K, G)
                c_score, _, _ = evaluate_consistency(K, G)
            else:
                G, R, c_score = None, None, 1.0

            # Floor Grid & Shadows in Viewport
            grid_2d, val_g = project_viewport_3d(grid_pts_3d, R_cam, T_cam, PANEL_W, VIEW_W)
            grid_2d_int = grid_2d.astype(np.int32)
            for i in range(0, len(grid_2d_int), 2):
                if val_g[i] and val_g[i+1]:
                    cv2.line(canvas, grid_2d_int[i], grid_2d_int[i+1], (42, 42, 52), 1)

            sh_pts = jp.copy(); sh_pts[:, 1] = Y_FLOOR
            sh_2d, val_sh = project_viewport_3d(sh_pts, R_cam, T_cam, PANEL_W, VIEW_W)
            sh_int = sh_2d.astype(np.int32)
            if np.all(val_sh):
                for c, p in PARENT_INDICES.items():
                    if p != -1: cv2.line(canvas, sh_int[p], sh_int[c], (10, 10, 14), 3)

            # Shaded MANO Mesh
            if ui_view_mode in [2, 3, 4] and G is not None:
                m_2d, val_m = project_viewport_3d(G.vertices, R_cam, T_cam, PANEL_W, VIEW_W)
                m_int = m_2d.astype(np.int32)
                cam_fnorm = G.face_normals @ R_cam.T
                front_i = np.where(cam_fnorm[:, 2] < -0.05)[0]
                mean_z = (G.vertices[G.faces[front_i, 0], 2] + G.vertices[G.faces[front_i, 1], 2] + G.vertices[G.faces[front_i, 2], 2]) / 3.0
                order = front_i[np.argsort(mean_z)[::-1]]

                for fi in order:
                    tri = G.faces[fi]
                    if val_m[tri[0]] and val_m[tri[1]] and val_m[tri[2]]:
                        diff = float(np.clip(-np.dot(G.face_normals[fi], light_dir), 0.0, 1.0))
                        face_col = (int(120 * (0.35 + 0.65 * diff)), int(145 * (0.35 + 0.65 * diff)), int(175 * (0.35 + 0.65 * diff)))
                        cv2.fillConvexPoly(canvas, m_int[tri], face_col)
                        cv2.polylines(canvas, [m_int[tri]], True, (max(0, face_col[0]-25), max(0, face_col[1]-25), max(0, face_col[2]-25)), 1)

            # Whiskers Overlay
            if ui_show_whiskers and R is not None and ui_view_mode == 4:
                w_pts = []
                for b_i, (p, c) in enumerate(BONES):
                    mid = (K.joint_positions[p] + K.joint_positions[c]) * 0.5
                    u, w = K.bone_frames[b_i, :, 1], K.bone_frames[b_i, :, 2]
                    for d_k, d_vec in enumerate([w, -w, u, -u]):
                        w_pts.extend([mid, mid + d_vec * R.bone_radial_depths[b_i, d_k]])
                w_2d, val_w = project_viewport_3d(np.array(w_pts), R_cam, T_cam, PANEL_W, VIEW_W)
                w_int = w_2d.astype(np.int32)
                for k in range(0, len(w_int), 2):
                    if val_w[k] and val_w[k+1]:
                        cv2.line(canvas, w_int[k], w_int[k+1], (0, 240, 200), 1)

            # Clearance Stalks Overlay
            if ui_show_clearance and R is not None and ui_view_mode == 4:
                v1 = K.joint_positions[5] - K.joint_positions[0]
                v2 = K.joint_positions[17] - K.joint_positions[0]
                pnorm = np.cross(v1, v2); pnorm /= max(np.linalg.norm(pnorm), 1e-9)
                for t_i, apex_id in enumerate(MANO_TIPS):
                    tip_pt = G.vertices[apex_id]
                    proj_pt = tip_pt - np.dot(tip_pt - K.joint_positions[0], pnorm) * pnorm
                    c_pts, val_c = project_viewport_3d(np.array([tip_pt, proj_pt]), R_cam, T_cam, PANEL_W, VIEW_W)
                    if np.all(val_c):
                        cv2.line(canvas, c_pts[0].astype(int), c_pts[1].astype(int), (0, 180, 255), 2)

            # Skeleton Overlay
            if ui_view_mode in [1, 3, 4]:
                k_2d, val_k = project_viewport_3d(jp, R_cam, T_cam, PANEL_W, VIEW_W)
                k_int = k_2d.astype(np.int32)
                for c, p in PARENT_INDICES.items():
                    if p != -1 and val_k[c] and val_k[p]:
                        cv2.line(canvas, k_int[p], k_int[c], (0, 220, 255) if ui_view_mode != 1 else (210, 210, 210), 2)
                for i in range(21):
                    if val_k[i]:
                        cv2.circle(canvas, k_int[i], 4, (0, 0, 255) if i == 0 else (40, 255, 40), -1)

            # Left Control Panel
            canvas[:, :PANEL_W] = (24, 24, 30)
            cv2.line(canvas, (PANEL_W - 1, 0), (PANEL_W - 1, CANVAS_H), (55, 55, 68), 2)
            cv2.putText(canvas, "K-G-R STUDIO", (22, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

            btn_col = (0, 70, 180) if is_auto_motion else (45, 45, 58)
            cv2.rectangle(canvas, (20, 80), (320, 115), btn_col, -1)
            cv2.putText(canvas, "[SPACE] AUTO MOTION: " + ("ON" if is_auto_motion else "OFF"), (38, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

            for idx, (k, lbl, mn, mx, def_v, is_i) in enumerate(SLIDER_SPECS):
                val = slider_vals[k]
                y_s, bx0, bx1 = get_slider_layout(idx)
                cv2.putText(canvas, lbl, (20, y_s + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 220) if not is_auto_motion else (110, 110, 130), 1)
                cv2.putText(canvas, f"{val:d}" if is_i else f"{val:.1f}", (105, y_s + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 180, 220), 1)
                cv2.line(canvas, (bx0, y_s), (bx1, y_s), (45, 45, 58), 4)
                frac = float(np.clip((val - mn) / float(mx - mn), 0.0, 1.0))
                cx = int(bx0 + frac * (bx1 - bx0))
                cv2.line(canvas, (bx0, y_s), (cx, y_s), (0, 180, 255) if not is_auto_motion else (70, 75, 85), 4)
                cv2.circle(canvas, (cx, y_s), 6, (255, 255, 255) if not is_auto_motion else (100, 105, 115), -1)

            w_col = (0, 140, 100) if ui_show_whiskers else (45, 45, 55)
            cv2.rectangle(canvas, (20, 655), (160, 685), w_col, -1)
            cv2.putText(canvas, f"Whiskers: {'ON' if ui_show_whiskers else 'OFF'}", (32, 674), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

            c_col = (0, 140, 100) if ui_show_clearance else (45, 45, 55)
            cv2.rectangle(canvas, (170, 655), (320, 685), c_col, -1)
            cv2.putText(canvas, f"Clearance: {'ON' if ui_show_clearance else 'OFF'}", (180, 674), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

            mode_labels = ["Skel", "MANO", "M+S", "K-G-R"]
            for b_i in range(1, 5):
                bx = 20 + (b_i - 1) * 75
                m_col = (0, 150, 220) if ui_view_mode == b_i else (40, 40, 52)
                cv2.rectangle(canvas, (bx, 710), (bx + 70, 742), m_col, -1)
                cv2.putText(canvas, mode_labels[b_i - 1], (bx + 14, 731), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

            cv2.putText(canvas, f"Phase: {phase_label}", (PANEL_W + 20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
            cv2.putText(canvas, f"Consistency C(K, G): {c_score:4.3f}", (PANEL_W + 20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)
            cv2.rectangle(canvas, (PANEL_W + 215, 84), (PANEL_W + 395, 98), (50, 50, 60), -1)
            cv2.rectangle(canvas, (PANEL_W + 215, 84), (PANEL_W + 215 + int(180 * c_score), 98), (0, 255, 100), -1)

        # -------------------------------------------------------------
        # TAB 2: SMOOTH MOTION & JITTER VIDEO TEST
        # -------------------------------------------------------------
        elif active_app_mode == 2:
            ctrl_in, phase_label = evaluate_smooth_choreography(sim_time)
            raw_state = state_update(persistent_state, ctrl_in, dt)
            persistent_state = project_constraints(raw_state)

            jp, jr = forward_kinematics(persistent_state)
            vm, fm, k21 = mano.forward(persistent_state, jp)

            if prev_mesh_verts is not None:
                disp = np.linalg.norm(vm - prev_mesh_verts, axis=1) * 1000.0
                max_jitter_mm = 0.85 * max_jitter_mm + 0.15 * float(np.max(disp))
            prev_mesh_verts = vm.copy()

            # Render Full Viewport
            grid_2d, val_g = project_viewport_3d(grid_pts_3d, R_cam, T_cam, 0, CANVAS_W)
            grid_int = grid_2d.astype(np.int32)
            for i in range(0, len(grid_int), 2):
                if val_g[i] and val_g[i+1]: cv2.line(canvas, grid_int[i], grid_int[i+1], (42, 42, 52), 1)

            m_2d, val_m = project_viewport_3d(vm, R_cam, T_cam, 0, CANVAS_W)
            m_int = m_2d.astype(np.int32)

            v0, v1, v2 = vm[fm[:, 0]], vm[fm[:, 1]], vm[fm[:, 2]]
            fnorm = np.cross(v1 - v0, v2 - v0); fnorm /= np.maximum(np.linalg.norm(fnorm, axis=1, keepdims=True), 1e-9)
            front_i = np.where((fnorm @ R_cam.T)[:, 2] < -0.05)[0]
            order = front_i[np.argsort((v0[front_i, 2] + v1[front_i, 2] + v2[front_i, 2]) / 3.0)[::-1]]

            for fi in order:
                tri = fm[fi]
                if val_m[tri[0]] and val_m[tri[1]] and val_m[tri[2]]:
                    diff = float(np.clip(-np.dot(fnorm[fi], light_dir), 0.0, 1.0))
                    face_col = (int(125 * (0.35 + 0.65 * diff)), int(150 * (0.35 + 0.65 * diff)), int(180 * (0.35 + 0.65 * diff)))
                    cv2.fillConvexPoly(canvas, m_int[tri], face_col)

            k_2d, val_k = project_viewport_3d(jp, R_cam, T_cam, 0, CANVAS_W)
            k_int = k_2d.astype(np.int32)
            for c, p in PARENT_INDICES.items():
                if p != -1 and val_k[c] and val_k[p]: cv2.line(canvas, k_int[p], k_int[c], (0, 220, 255), 2)

            rec_text = "[RECORDING MP4]" if is_recording else "[V] Toggle MP4 Record"
            rec_col = (0, 0, 255) if is_recording else (170, 170, 185)
            cv2.putText(canvas, f"SMOOTH CHOREOGRAPHY: {phase_label}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(canvas, f"Inter-Frame Mesh Jitter: {max_jitter_mm:.2f} mm/frame | {rec_text}", (30, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.46, rec_col, 1)

            if is_recording and video_writer is not None:
                video_writer.write(canvas)

        # -------------------------------------------------------------
        # TAB 3: 3-PANEL TEMPORAL PREDICTION & ERROR STUDIO
        # -------------------------------------------------------------
        elif active_app_mode == 3:
            N_samples = test_split.X_pred.shape[0]
            PANEL_3W = CANVAS_W // 3

            K_gt = test_split.K_gt[pred_frame_idx]
            G_gt = test_split.G_gt[pred_frame_idx]
            K_pr = test_split.K_pred[pred_frame_idx]
            G_pr = test_split.G_pred[pred_frame_idx]
            K_prev = test_split.K_prev[pred_frame_idx]
            G_prev = test_split.G_prev[pred_frame_idx]

            dK_rms = float(np.sqrt(np.mean((K_pr - K_gt) ** 2)) * 1000.0)
            dG_rms = float(np.sqrt(np.mean((G_pr - G_gt) ** 2)) * 1000.0)
            tips = np.linalg.norm(K_pr[[4, 8, 12, 16, 20]] - K_gt[[4, 8, 12, 16, 20]], axis=-1) * 1000.0

            cv2.line(canvas, (PANEL_3W, 42), (PANEL_3W, CANVAS_H), (50, 50, 60), 2)
            cv2.line(canvas, (2 * PANEL_3W, 42), (2 * PANEL_3W, CANVAS_H), (50, 50, 60), 2)

            def draw_3p_hand(K_pts, G_pts, p_idx, col_mesh, is_wireframe=False):
                m_2d, val_m = project_viewport_3d(G_pts, R_cam, T_cam, p_idx * PANEL_3W, PANEL_3W)
                m_int = m_2d.astype(np.int32)
                v0, v1, v2 = G_pts[mano.faces[:, 0]], G_pts[mano.faces[:, 1]], G_pts[mano.faces[:, 2]]
                fnorm = np.cross(v1 - v0, v2 - v0); fnorm /= np.maximum(np.linalg.norm(fnorm, axis=1, keepdims=True), 1e-9)
                front_i = np.where((fnorm @ R_cam.T)[:, 2] < -0.05)[0]
                order = front_i[np.argsort((v0[front_i, 2] + v1[front_i, 2] + v2[front_i, 2]) / 3.0)[::-1]]

                if not is_wireframe:
                    for fi in order[::2]:
                        tri = mano.faces[fi]
                        if val_m[tri[0]] and val_m[tri[1]] and val_m[tri[2]]:
                            diff = float(np.clip(-np.dot(fnorm[fi], light_dir), 0.0, 1.0))
                            face_col = (int(col_mesh[0] * (0.32 + 0.68 * diff)), int(col_mesh[1] * (0.32 + 0.68 * diff)), int(col_mesh[2] * (0.32 + 0.68 * diff)))
                            cv2.fillConvexPoly(canvas, m_int[tri], face_col)
                else:
                    for fi in range(0, len(mano.faces), 6):
                        tri = mano.faces[fi]
                        if val_m[tri[0]] and val_m[tri[1]] and val_m[tri[2]]: cv2.polylines(canvas, [m_int[tri]], True, col_mesh, 1)

                k_2d, val_k = project_viewport_3d(K_pts, R_cam, T_cam, p_idx * PANEL_3W, PANEL_3W)
                k_int = k_2d.astype(np.int32)
                for c, p in PARENT_INDICES.items():
                    if p != -1 and val_k[c] and val_k[p]: cv2.line(canvas, k_int[p], k_int[c], (0, 220, 255), 2)

            if not pred_timeline_mode:
                draw_3p_hand(K_gt, G_gt, 0, (120, 175, 120))
                cv2.putText(canvas, "PANEL 1: GROUND TRUTH (GT)", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 255, 150), 2)

                draw_3p_hand(K_pr, G_pr, 1, (175, 140, 110))
                cv2.putText(canvas, "PANEL 2: PREDICTED PRIOR (S_t^-)", (PANEL_3W + 20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 120), 2)

                draw_3p_hand(K_gt, G_gt, 2, (70, 70, 80), is_wireframe=True)
                k_gt_2d, _ = project_viewport_3d(K_gt, R_cam, T_cam, 2 * PANEL_3W, PANEL_3W)
                k_pr_2d, _ = project_viewport_3d(K_gt + (K_pr - K_gt) * pred_error_scale, R_cam, T_cam, 2 * PANEL_3W, PANEL_3W)
                for i in range(21):
                    cv2.line(canvas, k_gt_2d[i].astype(int), k_pr_2d[i].astype(int), (0, 70, 255), 2)
                cv2.putText(canvas, f"PANEL 3: ERROR OVERLAY ({pred_error_scale:.0f}x Scale)", (2 * PANEL_3W + 20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 150, 255), 2)
            else:
                draw_3p_hand(K_prev, G_prev, 0, (140, 140, 160))
                cv2.putText(canvas, "TIMELINE 1: S[t-1] (Previous)", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 220), 2)
                draw_3p_hand(K_pr, G_pr, 1, (180, 130, 90))
                cv2.putText(canvas, "TIMELINE 2: S[t]^- (Prior Prediction)", (PANEL_3W + 20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 180, 100), 2)
                draw_3p_hand(K_gt, G_gt, 2, (100, 180, 100))
                cv2.putText(canvas, "TIMELINE 3: S[t] (Ground Truth Target)", (2 * PANEL_3W + 20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 255, 120), 2)

            cv2.putText(canvas, f"Frame: {pred_frame_idx:04d} / {N_samples} | Delta_K RMS: {dK_rms:.2f} mm | Delta_G RMS: {dG_rms:.2f} mm | Mean Tip: {np.mean(tips):.2f} mm",
                        (20, CANVAS_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 120), 1)

            if not pred_is_paused:
                pred_frame_idx = (pred_frame_idx + 1) % N_samples

        render_top_navigation(canvas)

        cycle_dt = time.perf_counter() - t0
        fps_smooth = 0.92 * fps_smooth + 0.08 * (1.0 / max(cycle_dt, 1e-6))
        cv2.putText(canvas, f"FPS: {fps_smooth:3.0f}", (CANVAS_W - 90, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 255, 100), 1)

        cv2.imshow("K-G-R Unified State & Simulator Studio", canvas)

        wait_ms = 30 if (active_app_mode == 3 and pred_is_paused) else 16
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in [27, ord('q')]:
            break
        elif key == 0 or key == 190 or key == ord('1'):  # F1
            active_app_mode = 1
        elif key == 0 or key == 191 or key == ord('2'):  # F2
            active_app_mode = 2
        elif key == 0 or key == 192 or key == ord('3'):  # F3
            active_app_mode = 3
        elif key == ord(' '):
            if active_app_mode == 1: is_auto_motion = not is_auto_motion
            elif active_app_mode == 3: pred_is_paused = not pred_is_paused
        elif key in [ord('v'), ord('V')]:
            if active_app_mode == 2:
                is_recording = not is_recording
                if is_recording:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_path = ROOT / "experiments" / "smooth_motion_recording.mp4"
                    video_writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (CANVAS_W, CANVAS_H))
                elif video_writer is not None:
                    video_writer.release()
                    video_writer = None
        elif key in [ord('m'), ord('M')]:
            if active_app_mode == 3: pred_timeline_mode = not pred_timeline_mode
        elif key in [ord('n'), ord('N')]:
            if active_app_mode == 3: pred_frame_idx = (pred_frame_idx + 1) % test_split.X_pred.shape[0]
        elif key in [ord('e'), ord('E')]:
            if active_app_mode == 3: pred_error_scale = 1.0 if pred_error_scale == 5.0 else (pred_error_scale * 2.0)

    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_main()
