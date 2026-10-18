"""Stage 1: Explicit 3D Metric Spatial Simulation Space for Persistent Hand Tracker."""

import sys
from pathlib import Path
import importlib
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_config = importlib.import_module("00_spec.config")
_state = importlib.import_module("00_spec.state_definition")
_topology = importlib.import_module("00_spec.hand_topology")
_coord = importlib.import_module("00_spec.coordinate_system")
_const = importlib.import_module("02_state.constraints")
_fk = importlib.import_module("02_state.forward_kinematics")
_surface = importlib.import_module("02_state.surface")

HandState = _state.HandState
PARENT_INDICES = _topology.PARENT_INDICES
euler_to_quaternion = _coord.euler_to_quaternion
quaternion_to_rotation_matrix = _coord.quaternion_to_rotation_matrix
project_constraints = _const.project_constraints
forward_kinematics = _fk.forward_kinematics
generate_glove_surface = _surface.generate_glove_surface
compute_capsule_facets = _surface.compute_capsule_facets

# Canvas Dimensions
CANVAS_W = 900
CANVAS_H = 750
FOCAL_PX = 720.0

# Hand State Controls (World Coordinates)
ui_pos_x = 0        # mm
ui_pos_y = -30      # mm (Hand resting above floor)
ui_pos_z = 520      # mm
ui_pitch = 180      # deg (180 = 0)
ui_yaw   = 180      # deg
ui_roll  = 180      # deg
ui_splay = 15       # deg
ui_thumb_flex = 0   # 0 to 100
ui_thumb_abd  = 0   # 0 to 100
ui_index_bend = 0
ui_middle_bend= 0
ui_ring_bend  = 0
ui_pinky_bend = 0
ui_view_mode  = 3   # 1: Skeleton, 2: Glove, 3: Hybrid

# 3D Camera Orbit Controls
cam_orbit_yaw = 0.0     # rad
cam_orbit_pitch = 0.20  # rad (slight top-down view)
is_orbiting_cam = False

# Mouse Interaction Flags
is_dragging_xy = False
is_rotating_hand = False
is_depth_roll = False
last_mouse_x = 0
last_mouse_y = 0

def project_3d_to_screen(pts_3d: np.ndarray, R_cam: np.ndarray, T_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Applies camera transform and perspective projection."""
    pts = np.atleast_2d(pts_3d)
    pts_cam = (R_cam @ (pts - T_cam).T).T
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    valid = Z > 0.04
    safe_z = np.where(valid, Z, 1.0)
    u = (FOCAL_PX * (X / safe_z)) + (CANVAS_W / 2.0)
    v = (FOCAL_PX * (Y / safe_z)) + (CANVAS_H / 2.0)
    return np.column_stack([u, v]), valid

def on_mouse_event(event, x, y, flags, userdata):
    global is_dragging_xy, is_rotating_hand, is_depth_roll, is_orbiting_cam
    global last_mouse_x, last_mouse_y
    global ui_pos_x, ui_pos_y, ui_pos_z, ui_pitch, ui_yaw, ui_roll
    global cam_orbit_yaw, cam_orbit_pitch

    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            is_depth_roll = True
        elif flags & cv2.EVENT_FLAG_CTRLKEY:
            is_orbiting_cam = True
        else:
            is_dragging_xy = True
        last_mouse_x, last_mouse_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging_xy = False
        is_depth_roll = False
        is_orbiting_cam = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY or flags & cv2.EVENT_FLAG_CTRLKEY:
            is_orbiting_cam = True
        else:
            is_rotating_hand = True
        last_mouse_x, last_mouse_y = x, y

    elif event == cv2.EVENT_RBUTTONUP:
        is_rotating_hand = False
        is_orbiting_cam = False

    elif event == cv2.EVENT_MBUTTONDOWN:
        is_orbiting_cam = True
        last_mouse_x, last_mouse_y = x, y

    elif event == cv2.EVENT_MBUTTONUP:
        is_orbiting_cam = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        delta = 30 if flags > 0 else -30
        ui_pos_z = int(np.clip(ui_pos_z - delta, 250, 1400))
        cv2.setTrackbarPos("Pos Z (Depth)", "Controls", ui_pos_z)

    elif event == cv2.EVENT_MOUSEMOVE:
        dx = x - last_mouse_x
        dy = y - last_mouse_y

        if is_orbiting_cam:
            cam_orbit_yaw -= dx * 0.008
            cam_orbit_pitch = float(np.clip(cam_orbit_pitch - dy * 0.008, -0.65, 0.85))
            last_mouse_x, last_mouse_y = x, y

        elif is_dragging_xy:
            ui_pos_x = int(np.clip(ui_pos_x + dx, -350, 350))
            ui_pos_y = int(np.clip(ui_pos_y + dy, -300, 200))
            cv2.setTrackbarPos("Pos X", "Controls", ui_pos_x + 350)
            cv2.setTrackbarPos("Pos Y", "Controls", ui_pos_y + 300)
            last_mouse_x, last_mouse_y = x, y

        elif is_rotating_hand:
            ui_yaw = (ui_yaw + dx + 360) % 360
            ui_pitch = (ui_pitch - dy + 360) % 360
            cv2.setTrackbarPos("Yaw (Turn)", "Controls", ui_yaw)
            cv2.setTrackbarPos("Pitch (Tilt)", "Controls", ui_pitch)
            last_mouse_x, last_mouse_y = x, y

        elif is_depth_roll:
            ui_roll = (ui_roll + dx + 360) % 360
            ui_pos_z = int(np.clip(ui_pos_z + dy * 2, 250, 1400))
            cv2.setTrackbarPos("Roll (Twist)", "Controls", ui_roll)
            cv2.setTrackbarPos("Pos Z (Depth)", "Controls", ui_pos_z)
            last_mouse_x, last_mouse_y = x, y

def reset_scene():
    global ui_pos_x, ui_pos_y, ui_pos_z, ui_pitch, ui_yaw, ui_roll
    global ui_splay, ui_thumb_flex, ui_thumb_abd, ui_index_bend, ui_middle_bend, ui_ring_bend, ui_pinky_bend
    global cam_orbit_yaw, cam_orbit_pitch
    ui_pos_x = 0
    ui_pos_y = -30
    ui_pos_z = 520
    ui_pitch = 180
    ui_yaw   = 180
    ui_roll  = 180
    ui_splay = 15
    ui_thumb_flex = 0
    ui_thumb_abd  = 0
    ui_index_bend = 0
    ui_middle_bend= 0
    ui_ring_bend  = 0
    ui_pinky_bend = 0
    cam_orbit_yaw = 0.0
    cam_orbit_pitch = 0.20

    cv2.setTrackbarPos("Pos X", "Controls", ui_pos_x + 350)
    cv2.setTrackbarPos("Pos Y", "Controls", ui_pos_y + 300)
    cv2.setTrackbarPos("Pos Z (Depth)", "Controls", ui_pos_z)
    cv2.setTrackbarPos("Pitch (Tilt)", "Controls", ui_pitch)
    cv2.setTrackbarPos("Yaw (Turn)", "Controls", ui_yaw)
    cv2.setTrackbarPos("Roll (Twist)", "Controls", ui_roll)
    cv2.setTrackbarPos("Splay", "Controls", ui_splay)
    cv2.setTrackbarPos("Thumb Flex", "Controls", ui_thumb_flex)
    cv2.setTrackbarPos("Thumb Abd", "Controls", ui_thumb_abd)
    cv2.setTrackbarPos("Index Curl", "Controls", ui_index_bend)
    cv2.setTrackbarPos("Middle Curl", "Controls", ui_middle_bend)
    cv2.setTrackbarPos("Ring Curl", "Controls", ui_ring_bend)
    cv2.setTrackbarPos("Pinky Curl", "Controls", ui_pinky_bend)

def dummy(x): pass

def run_3d_spatial_simulator():
    global ui_pos_x, ui_pos_y, ui_pos_z, ui_pitch, ui_yaw, ui_roll
    global ui_splay, ui_thumb_flex, ui_thumb_abd, ui_index_bend, ui_middle_bend, ui_ring_bend, ui_pinky_bend, ui_view_mode

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 360, 560)

    cv2.createTrackbar("Pos X", "Controls", ui_pos_x + 350, 700, dummy)
    cv2.createTrackbar("Pos Y", "Controls", ui_pos_y + 300, 500, dummy)
    cv2.createTrackbar("Pos Z (Depth)", "Controls", ui_pos_z, 1400, dummy)
    cv2.createTrackbar("Pitch (Tilt)", "Controls", ui_pitch, 360, dummy)
    cv2.createTrackbar("Yaw (Turn)", "Controls", ui_yaw, 360, dummy)
    cv2.createTrackbar("Roll (Twist)", "Controls", ui_roll, 360, dummy)
    cv2.createTrackbar("Splay", "Controls", ui_splay, 45, dummy)
    cv2.createTrackbar("Thumb Flex", "Controls", ui_thumb_flex, 100, dummy)
    cv2.createTrackbar("Thumb Abd", "Controls", ui_thumb_abd, 100, dummy)
    cv2.createTrackbar("Index Curl", "Controls", ui_index_bend, 100, dummy)
    cv2.createTrackbar("Middle Curl", "Controls", ui_middle_bend, 100, dummy)
    cv2.createTrackbar("Ring Curl", "Controls", ui_ring_bend, 100, dummy)
    cv2.createTrackbar("Pinky Curl", "Controls", ui_pinky_bend, 100, dummy)
    cv2.createTrackbar("View Mode", "Controls", ui_view_mode, 3, dummy)

    cv2.namedWindow("Explicit 3D Spatial Simulator", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Explicit 3D Spatial Simulator", on_mouse_event)

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    persistent_state = HandState.neutral(z_depth=0.52)

    # 3D Environment Definition
    Y_FLOOR = 0.20  # Floor plane at +0.20m (meters down)
    grid_x = np.linspace(-0.35, 0.35, 15)
    grid_z = np.linspace(0.20, 0.85, 14)

    # Directional Light Source (Normalized)
    light_dir = np.array([0.4, -0.7, -0.6], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)

    print("\n=======================================================")
    print(" EXPLICIT 3D HAND SIMULATION SPACE ENGAGED")
    print("-------------------------------------------------------")
    print(" Left-Drag              : Move Hand (X / Y in 3D Space)")
    print(" Right-Drag             : Rotate Hand (Pitch / Yaw)")
    print(" Scroll Wheel           : Move Hand Depth (Z)")
    print(" Shift + Left-Drag      : Depth (Z) & Twist (Roll)")
    print(" Middle-Drag / Ctrl-Drag: Orbit 3D Camera Around Scene")
    print(" 'R' Key                : Reset All Poses to Neutral")
    print(" '1' / '2' / '3' Keys   : Toggle View Modes")
    print("=======================================================\n")

    fps_smooth = 60.0

    while True:
        t0 = time.perf_counter()

        # Ingest trackbars
        ui_pos_x = cv2.getTrackbarPos("Pos X", "Controls") - 350
        ui_pos_y = cv2.getTrackbarPos("Pos Y", "Controls") - 300
        ui_pos_z = max(200, cv2.getTrackbarPos("Pos Z (Depth)", "Controls"))
        ui_pitch = cv2.getTrackbarPos("Pitch (Tilt)", "Controls")
        ui_yaw   = cv2.getTrackbarPos("Yaw (Turn)", "Controls")
        ui_roll  = cv2.getTrackbarPos("Roll (Twist)", "Controls")
        ui_splay = cv2.getTrackbarPos("Splay", "Controls")
        ui_thumb_flex = cv2.getTrackbarPos("Thumb Flex", "Controls")
        ui_thumb_abd  = cv2.getTrackbarPos("Thumb Abd", "Controls")
        ui_index_bend = cv2.getTrackbarPos("Index Curl", "Controls")
        ui_middle_bend= cv2.getTrackbarPos("Middle Curl", "Controls")
        ui_ring_bend  = cv2.getTrackbarPos("Ring Curl", "Controls")
        ui_pinky_bend = cv2.getTrackbarPos("Pinky Curl", "Controls")
        ui_view_mode  = max(1, cv2.getTrackbarPos("View Mode", "Controls"))

        # 1. Update Persistent Hand State (meters & radians in World Space)
        persistent_state.T[0] = ui_pos_x / 1000.0
        persistent_state.T[1] = ui_pos_y / 1000.0
        persistent_state.T[2] = ui_pos_z / 1000.0

        r_rad = np.radians(ui_roll - 180)
        p_rad = np.radians(ui_pitch - 180)
        y_rad = np.radians(ui_yaw - 180)
        persistent_state.R = euler_to_quaternion(r_rad, p_rad, y_rad)

        splay_rad = np.radians(ui_splay)
        persistent_state.q[0] = (ui_thumb_abd / 100.0) * 0.6
        persistent_state.q[1] = (ui_thumb_flex / 100.0) * 0.5
        persistent_state.q[2] = (ui_thumb_abd / 100.0) * 0.3
        persistent_state.q[3] = (ui_thumb_flex / 100.0) * 0.8
        persistent_state.q[4] = (ui_thumb_flex / 100.0) * 1.2

        curls = [
            (ui_index_bend / 100.0, splay_rad * 1.2, 5),
            (ui_middle_bend / 100.0, splay_rad * 0.2, 9),
            (ui_ring_bend / 100.0, -splay_rad * 0.4, 13),
            (ui_pinky_bend / 100.0, -splay_rad * 1.2, 17)
        ]
        for c_val, s_val, q_idx in curls:
            persistent_state.q[q_idx]     = s_val
            persistent_state.q[q_idx + 1] = c_val * 1.45
            persistent_state.q[q_idx + 2] = c_val * 1.65
            persistent_state.q[q_idx + 3] = c_val * 1.30

        # 2. Physics Constraints & Forward Kinematics
        state_valid = project_constraints(persistent_state)
        joint_pos_3d, _ = forward_kinematics(state_valid)

        # 3. Setup Dynamic Orbiting 3D Camera Frame
        # Scene Center Target
        scene_center = np.array([0.0, -0.02, 0.52])
        # Camera Extrinsic Matrices
        cy, sy = np.cos(cam_orbit_yaw), np.sin(cam_orbit_yaw)
        cp, sp = np.cos(cam_orbit_pitch), np.sin(cam_orbit_pitch)

        # Orbit Rotation Matrix
        R_orbit_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
        R_orbit_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
        R_cam = R_orbit_x @ R_orbit_y
        # Camera origin in World
        T_cam = scene_center - R_cam.T @ np.array([0.0, 0.0, 0.52])

        # Project Hand Nodes to Camera Space
        pts_2d, valid_nodes = project_3d_to_screen(joint_pos_3d, R_cam, T_cam)

        # 4. Rasterization Pipeline (Painter's Depth Sorting)
        canvas[:] = (22, 22, 28)  # Deep Charcoal 3D Space Background

        render_queue = []  # List of tuples: (depth_z, type, data)

        # ── 4A. 3D Spatial Ground Grid ──
        # Longitudinal Lines (parallel to Z)
        for gx in grid_x:
            p_start = np.array([gx, Y_FLOOR, grid_z[0]])
            p_end   = np.array([gx, Y_FLOOR, grid_z[-1]])
            pts_line, val = project_3d_to_screen(np.array([p_start, p_end]), R_cam, T_cam)
            if np.all(val):
                mid_z = (p_start[2] + p_end[2]) * 0.5
                render_queue.append((mid_z, 'GRID_LINE', (pts_line[0].astype(np.int32), pts_line[1].astype(np.int32))))

        # Lateral Lines (parallel to X)
        for gz in grid_z:
            p_start = np.array([grid_x[0], Y_FLOOR, gz])
            p_end   = np.array([grid_x[-1], Y_FLOOR, gz])
            pts_line, val = project_3d_to_screen(np.array([p_start, p_end]), R_cam, T_cam)
            if np.all(val):
                render_queue.append((gz, 'GRID_LINE', (pts_line[0].astype(np.int32), pts_line[1].astype(np.int32))))

        # ── 4B. World Origin Triad (Anchored on Ground at 0, Y_floor, 0.52) ──
        origin_w = np.array([0.0, Y_FLOOR, 0.52])
        axis_len = 0.08
        axes_w = [
            (origin_w + np.array([axis_len, 0, 0]), (0, 0, 255), "X_w"),      # Red X
            (origin_w + np.array([0, -axis_len, 0]), (0, 255, 0), "Y_w (Up)"), # Green -Y (Up in 3D)
            (origin_w + np.array([0, 0, axis_len]), (255, 120, 0), "Z_w")     # Blue Z
        ]
        pt_orig_2d, v_o = project_3d_to_screen(origin_w, R_cam, T_cam)
        if v_o[0]:
            for end_w, col, label in axes_w:
                pt_end_2d, v_e = project_3d_to_screen(end_w, R_cam, T_cam)
                if v_e[0]:
                    render_queue.append((origin_w[2], 'AXIS', (pt_orig_2d[0].astype(np.int32), pt_end_2d[0].astype(np.int32), col, label)))

        # ── 4C. 3D Drop-Shadows and Height Stalks on the Ground Plane ──
        shadow_pts_3d = joint_pos_3d.copy()
        shadow_pts_3d[:, 1] = Y_FLOOR
        shadow_pts_2d, val_sh = project_3d_to_screen(shadow_pts_3d, R_cam, T_cam)

        if np.all(val_sh):
            # Shadow Bones
            for child, parent in PARENT_INDICES.items():
                if parent != -1:
                    mid_z = (shadow_pts_3d[parent, 2] + shadow_pts_3d[child, 2]) * 0.5
                    render_queue.append((mid_z, 'SHADOW_BONE', (shadow_pts_2d[parent].astype(np.int32), shadow_pts_2d[child].astype(np.int32))))

            # Altitude Stalks (Wrist + 5 Fingertips: 0, 4, 8, 12, 16, 20)
            key_nodes = [0, 4, 8, 12, 16, 20]
            for kn in key_nodes:
                render_queue.append((joint_pos_3d[kn, 2], 'STALK', (pts_2d[kn].astype(np.int32), shadow_pts_2d[kn].astype(np.int32))))

        # ── 4D. Wrist Local Orientation Gizmo ──
        R_wrist = quaternion_to_rotation_matrix(state_valid.R)
        wrist_pos = joint_pos_3d[0]
        gizmo_len = 0.045
        for ax_idx, ax_col in enumerate([(0, 0, 255), (0, 255, 0), (255, 120, 0)]):
            ax_end_3d = wrist_pos + R_wrist[:, ax_idx] * gizmo_len
            pts_ax, v_ax = project_3d_to_screen(np.array([wrist_pos, ax_end_3d]), R_cam, T_cam)
            if np.all(v_ax):
                render_queue.append((wrist_pos[2], 'LOCAL_AXIS', (pts_ax[0].astype(np.int32), pts_ax[1].astype(np.int32), ax_col)))

        # ── 4E. Volumetric 3D Glove Mesh Facets ──
        if ui_view_mode in [2, 3]:
            capsules = generate_glove_surface(joint_pos_3d)
            for cap in capsules:
                facets = compute_capsule_facets(cap, num_facets=8)
                for quad_3d, normal, mean_z in facets:
                    # Directional Shading
                    diffuse = float(np.clip(np.dot(normal, -light_dir), 0.0, 1.0))
                    intensity = 0.28 + 0.72 * diffuse
                    base_bgr = np.array([180, 150, 125], dtype=np.float64)  # Cyan-Slate
                    face_col = np.clip(base_bgr * intensity, 0, 255).astype(np.uint8).tolist()

                    quad_2d, q_val = project_3d_to_screen(quad_3d, R_cam, T_cam)
                    if np.all(q_val):
                        render_queue.append((mean_z, 'FACET', (quad_2d.astype(np.int32), face_col)))

        # ── 4F. Skeleton Joints and Bones ──
        if ui_view_mode in [1, 3]:
            for child, parent in PARENT_INDICES.items():
                if parent != -1:
                    mid_z = (joint_pos_3d[parent, 2] + joint_pos_3d[child, 2]) * 0.5
                    render_queue.append((mid_z, 'BONE', (pts_2d[parent].astype(np.int32), pts_2d[child].astype(np.int32))))

            for i in range(21):
                render_queue.append((joint_pos_3d[i, 2], 'NODE', (pts_2d[i].astype(np.int32), i, joint_pos_3d[i, 2])))

        # Depth-Sort All Primitives (Back to Front)
        render_queue.sort(key=lambda item: item[0], reverse=True)

        # ── 5. Render Primitives ──
        for _, ptype, data in render_queue:
            if ptype == 'GRID_LINE':
                p1, p2 = data
                cv2.line(canvas, p1, p2, (45, 45, 55), 1, cv2.LINE_AA)

            elif ptype == 'SHADOW_BONE':
                p1, p2 = data
                cv2.line(canvas, p1, p2, (10, 10, 12), 4, cv2.LINE_AA)

            elif ptype == 'STALK':
                p_hand, p_floor = data
                cv2.line(canvas, p_hand, p_floor, (60, 80, 100), 1, cv2.LINE_4)
                cv2.circle(canvas, p_floor, 3, (15, 15, 20), -1, cv2.LINE_AA)

            elif ptype == 'AXIS':
                p1, p2, col, label = data
                cv2.arrowedLine(canvas, p1, p2, col, 2, cv2.LINE_AA, tipLength=0.25)
                cv2.putText(canvas, label, (p2[0] + 4, p2[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

            elif ptype == 'LOCAL_AXIS':
                p1, p2, col = data
                cv2.line(canvas, p1, p2, col, 2, cv2.LINE_AA)

            elif ptype == 'FACET':
                pts_poly, col = data
                cv2.fillConvexPoly(canvas, pts_poly, col, lineType=cv2.LINE_AA)
                cv2.polylines(canvas, [pts_poly], True, (max(0, col[0]-35), max(0, col[1]-35), max(0, col[2]-35)), 1, cv2.LINE_AA)

            elif ptype == 'BONE':
                p1, p2 = data
                b_color = (0, 220, 255) if ui_view_mode == 3 else (210, 210, 210)
                cv2.line(canvas, p1, p2, b_color, 2, cv2.LINE_AA)

            elif ptype == 'NODE':
                pt, idx, z_depth = data
                radius = int(np.clip(2600.0 / (z_depth * 1000.0 + 100.0), 3, 13))
                if idx == 0:
                    ncol = (0, 0, 255)     # Wrist: Red
                elif idx % 4 == 0:
                    ncol = (40, 255, 40)   # Tips: Bright Green
                else:
                    ncol = (0, 180, 255)   # Phalanx joints: Gold
                cv2.circle(canvas, pt, radius, ncol, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, radius, (255, 255, 255), 1, cv2.LINE_AA)

        # ── 6. Heads-Up Display (HUD) ──
        dt_frame = time.perf_counter() - t0
        latency_us = dt_frame * 1e6
        fps_curr = 1.0 / max(dt_frame, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps_curr

        mode_str = {1: "Skeleton Only", 2: "Volumetric Glove", 3: "Hybrid (Glove + Skeleton)"}[ui_view_mode]

        cv2.putText(canvas, "3D SPATIAL HAND & GLOVE SIMULATOR (STAGE 1)", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Performance: {fps_smooth:4.0f} FPS | Frame Latency: {latency_us:4.0f} us", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Wrist 3D Position : X={state_valid.T[0]:+.3f}m, Y={state_valid.T[1]:+.3f}m, Z={state_valid.T[2]:+.3f}m", (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Mode: {mode_str} (Keys 1-3) | Cam Orbit Yaw: {np.degrees(cam_orbit_yaw):+.1f} deg", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(canvas, "L-Drag: Move XY | R-Drag: Rotate Hand | Wheel: Depth | Middle/Ctrl-Drag: Orbit 3D Cam | R: Reset",
                    (20, CANVAS_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow("Explicit 3D Spatial Simulator", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('r'):
            reset_scene()
        elif key in [ord('1'), ord('2'), ord('3')]:
            ui_view_mode = int(chr(key))
            cv2.setTrackbarPos("View Mode", "Controls", ui_view_mode)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_3d_spatial_simulator()
