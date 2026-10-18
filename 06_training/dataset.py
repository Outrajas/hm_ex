"""Large-scale continuous trajectory generator with live progress and caching."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List
import time
import numpy as np
import importlib

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.import_module("00_spec.state_definition")
_coord = importlib.import_module("00_spec.coordinate_system")
_fk = importlib.import_module("02_state.forward_kinematics")
_mano = importlib.import_module("02_state.mano_surface")
_dyn = importlib.import_module("02_state.dynamics")
_rel = importlib.import_module("03_observation.relationship")
_state_engine = importlib.import_module("02_state.state")
_const = importlib.import_module("02_state.constraints")

HandState = _spec.HandState
forward_kinematics = _fk.forward_kinematics
ManoSurface = _mano.ManoSurface
predict_kinematics = _dyn.predict_kinematics
compute_skeleton_geometry = _rel.compute_skeleton_geometry
compute_glove_geometry = _rel.compute_glove_geometry
evaluate_relationship = _rel.evaluate_relationship
ControlInput = _state_engine.ControlInput
state_update = _state_engine.state_update
project_constraints = _const.project_constraints
euler_to_quaternion = _coord.euler_to_quaternion

CACHE_DIR = ROOT / "experiments" / "dataset_cache"

@dataclass
class TrajectoryTransitionBatch:
    X_prev: np.ndarray
    X_pred: np.ndarray
    X_gt: np.ndarray
    K_prev: np.ndarray
    K_pred: np.ndarray
    K_gt: np.ndarray
    G_prev: np.ndarray
    G_pred: np.ndarray
    G_gt: np.ndarray
    R_prev: np.ndarray
    R_pred: np.ndarray
    R_gt: np.ndarray
    delta_X: np.ndarray
    delta_q: np.ndarray
    delta_K: np.ndarray
    delta_G: np.ndarray
    trajectory_ids: np.ndarray

def _generate_single_trajectory(
    traj_id: int,
    n_frames: int,
    dt: float,
    beta: np.ndarray,
    motion_regime: str,
    mano: ManoSurface,
    rng: np.random.RandomState
):
    init_state = HandState.neutral(z_depth=rng.uniform(0.42, 0.65))
    init_state.beta = beta.copy()
    init_state.T[0] = rng.uniform(-0.15, 0.15)
    init_state.T[1] = rng.uniform(-0.10, 0.08)

    yaw = np.pi + rng.uniform(-0.4, 0.4)
    pitch = rng.uniform(-0.35, 0.35)
    roll = rng.uniform(-0.35, 0.35)
    init_state.R = euler_to_quaternion(roll, pitch, yaw)

    init_curl = rng.uniform(0.0, 0.8)
    init_state.q[0] = init_curl * 0.4
    init_state.q[1] = init_curl * 0.5
    for idx in [5, 9, 13, 17]:
        init_state.q[idx+1] = init_curl * rng.uniform(1.1, 1.5)
        init_state.q[idx+2] = init_curl * rng.uniform(1.2, 1.6)
        init_state.q[idx+3] = init_curl * rng.uniform(0.8, 1.2)

    init_state = project_constraints(init_state)

    n_harmonics = 3
    omega_base = rng.uniform(0.8, 3.5) if motion_regime in ["fast", "abrupt"] else rng.uniform(0.3, 1.4)
    amplitudes = rng.uniform(0.1, 0.7, size=(21, n_harmonics))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(21, n_harmonics))

    pos_drift = rng.uniform(-0.06, 0.06, size=3)
    rot_drift = rng.uniform(-0.4, 0.4, size=3)

    curr_state = init_state.copy()

    # Pre-evaluate initial frame to avoid redundant recalculation
    jp_init, jr_init = forward_kinematics(curr_state)
    last_K = compute_skeleton_geometry(jp_init, jr_init)
    vp_init, fp_init, k21_init = mano.forward(curr_state, jp_init)
    last_G = compute_glove_geometry(vp_init, fp_init, k21_init)
    last_R = evaluate_relationship(last_K, last_G)

    Xp_l, Xpr_l, Xg_l = [], [], []
    Kp_l, Kpr_l, Kg_l = [], [], []
    Gp_l, Gpr_l, Gg_l = [], [], []
    Rp_l, Rpr_l, Rg_l = [], [], []
    tid_l = []

    for f in range(n_frames):
        prev_state = curr_state.copy()
        t = f * dt

        target_q = init_state.q.copy()
        for d in range(21):
            h_sum = sum(amplitudes[d, k] * np.sin((k + 1) * omega_base * t + phases[d, k]) for k in range(n_harmonics))
            target_q[d] += h_sum * (0.8 if motion_regime != "finger_specific" else (1.4 if d in [6,7,8] else 0.1))

        if motion_regime == "fist":
            target_q[6:9] = 1.6; target_q[10:13] = 1.6; target_q[14:17] = 1.6; target_q[18:21] = 1.4
        elif motion_regime == "abrupt" and f == n_frames // 2:
            target_q += rng.uniform(-0.5, 0.5, size=21)

        target_T = init_state.T + pos_drift * np.sin(omega_base * 0.5 * t)
        target_R = euler_to_quaternion(
            roll + rot_drift[0] * np.sin(omega_base * 0.6 * t),
            pitch + rot_drift[1] * np.cos(omega_base * 0.6 * t),
            yaw + rot_drift[2] * np.sin(omega_base * 0.4 * t)
        )

        ctrl = ControlInput(
            target_T=target_T, target_R=target_R, target_q=target_q,
            authority=0.92 if motion_regime == "abrupt" else 0.75
        )

        raw_gt = state_update(prev_state, ctrl, dt)
        gt_state = project_constraints(raw_gt)
        gt_state.beta = beta.copy()

        pred_state = predict_kinematics(prev_state, dt)
        pred_state.beta = beta.copy()

        # Previous geometry (reused from preceding step)
        Kp = last_K
        Gp = last_G
        Rp = last_R

        # Predicted geometry
        jp_pr, jr_pr = forward_kinematics(pred_state)
        Kpr = compute_skeleton_geometry(jp_pr, jr_pr)
        vp_pr, fp_pr, k21_pr = mano.forward(pred_state, jp_pr)
        Gpr = compute_glove_geometry(vp_pr, fp_pr, k21_pr)
        Rpr = evaluate_relationship(Kpr, Gpr)

        # Ground truth geometry
        jp_g, jr_g = forward_kinematics(gt_state)
        Kg = compute_skeleton_geometry(jp_g, jr_g)
        vp_g, fp_g, k21_g = mano.forward(gt_state, jp_g)
        Gg = compute_glove_geometry(vp_g, fp_g, k21_g)
        Rg = evaluate_relationship(Kg, Gg)

        Xp_l.append(prev_state.to_vector())
        Xpr_l.append(pred_state.to_vector())
        Xg_l.append(gt_state.to_vector())

        Kp_l.append(Kp.joint_positions)
        Kpr_l.append(Kpr.joint_positions)
        Kg_l.append(Kg.joint_positions)

        Gp_l.append(Gp.vertices)
        Gpr_l.append(Gpr.vertices)
        Gg_l.append(Gg.vertices)

        Rp_l.append(Rp.to_vector())
        Rpr_l.append(Rpr.to_vector())
        Rg_l.append(Rg.to_vector())
        tid_l.append(traj_id)

        curr_state = gt_state.copy()
        last_K, last_G, last_R = Kg, Gg, Rg

    return Xp_l, Xpr_l, Xg_l, Kp_l, Kpr_l, Kg_l, Gp_l, Gpr_l, Gg_l, Rp_l, Rpr_l, Rg_l, tid_l

def build_split_dataset(
    n_trajectories: int = 600,
    frames_per_traj: int = 50,
    dt: float = 0.016,
    seed: int = 101,
    force_rebuild: bool = False
) -> Dict[str, TrajectoryTransitionBatch]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"trajectories_n{n_trajectories}_f{frames_per_traj}_s{seed}.npz"

    if cache_file.exists() and not force_rebuild:
        print(f"[Dataset] Loading cached trajectory transition bundle from {cache_file.name}...")
        raw = np.load(cache_file)
        splits = {}
        for sp in ["train", "val", "test", "unseen_beta"]:
            splits[sp] = TrajectoryTransitionBatch(
                X_prev=raw[f"{sp}_X_prev"], X_pred=raw[f"{sp}_X_pred"], X_gt=raw[f"{sp}_X_gt"],
                K_prev=raw[f"{sp}_K_prev"], K_pred=raw[f"{sp}_K_pred"], K_gt=raw[f"{sp}_K_gt"],
                G_prev=raw[f"{sp}_G_prev"], G_pred=raw[f"{sp}_G_pred"], G_gt=raw[f"{sp}_G_gt"],
                R_prev=raw[f"{sp}_R_prev"], R_pred=raw[f"{sp}_R_pred"], R_gt=raw[f"{sp}_R_gt"],
                delta_X=raw[f"{sp}_delta_X"], delta_q=raw[f"{sp}_delta_q"],
                delta_K=raw[f"{sp}_delta_K"], delta_G=raw[f"{sp}_delta_G"],
                trajectory_ids=raw[f"{sp}_traj_ids"]
            )
        return splits

    print(f"[Dataset] Synthesizing {n_trajectories} continuous trajectories ({n_trajectories * frames_per_traj} transitions)...")
    rng = np.random.RandomState(seed)
    mano = ManoSurface()

    n_train_t = int(n_trajectories * 0.60)
    n_val_t = int(n_trajectories * 0.15)
    n_test_t = int(n_trajectories * 0.15)
    n_unseen_t = n_trajectories - (n_train_t + n_val_t + n_test_t)

    regimes = ["normal", "slow", "fast", "abrupt", "finger_specific", "fist"]
    split_accum = {sp: [[],[],[], [],[],[], [],[],[], [],[],[], []] for sp in ["train", "val", "test", "unseen_beta"]}

    traj_id = 0
    t_start = time.time()

    for split_name, count, is_unseen in [
        ("train", n_train_t, False),
        ("val", n_val_t, False),
        ("test", n_test_t, False),
        ("unseen_beta", n_unseen_t, True)
    ]:
        for _ in range(count):
            regime = rng.choice(regimes)
            beta = rng.normal(0.0, 2.2, size=10) if is_unseen else rng.normal(0.0, 1.0, size=10)
            res = _generate_single_trajectory(traj_id, frames_per_traj, dt, beta, regime, mano, rng)
            for k in range(13):
                split_accum[split_name][k].extend(res[k])
            traj_id += 1

            if traj_id % 25 == 0 or traj_id == n_trajectories:
                elapsed = time.time() - t_start
                rate = elapsed / traj_id
                eta = rate * (n_trajectories - traj_id)
                pct = (traj_id / n_trajectories) * 100.0
                print(f"[Dataset] Trajectory {traj_id:3d}/{n_trajectories} ({pct:4.1f}%) | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    save_dict = {}
    splits = {}
    for sp in ["train", "val", "test", "unseen_beta"]:
        acc = split_accum[sp]
        Xp, Xpr, Xg = np.array(acc[0]), np.array(acc[1]), np.array(acc[2])
        Kp, Kpr, Kg = np.array(acc[3]), np.array(acc[4]), np.array(acc[5])
        Gp, Gpr, Gg = np.array(acc[6]), np.array(acc[7]), np.array(acc[8])
        Rp, Rpr, Rg = np.array(acc[9]), np.array(acc[10]), np.array(acc[11])
        tids = np.array(acc[12])

        splits[sp] = TrajectoryTransitionBatch(
            X_prev=Xp, X_pred=Xpr, X_gt=Xg,
            K_prev=Kp, K_pred=Kpr, K_gt=Kg,
            G_prev=Gp, G_pred=Gpr, G_gt=Gg,
            R_prev=Rp, R_pred=Rpr, R_gt=Rg,
            delta_X=Xg - Xpr, delta_q=Xg[:, 7:28] - Xpr[:, 7:28],
            delta_K=Kg - Kpr, delta_G=Gg - Gpr,
            trajectory_ids=tids
        )

        save_dict[f"{sp}_X_prev"] = Xp; save_dict[f"{sp}_X_pred"] = Xpr; save_dict[f"{sp}_X_gt"] = Xg
        save_dict[f"{sp}_K_prev"] = Kp; save_dict[f"{sp}_K_pred"] = Kpr; save_dict[f"{sp}_K_gt"] = Kg
        save_dict[f"{sp}_G_prev"] = Gp; save_dict[f"{sp}_G_pred"] = Gpr; save_dict[f"{sp}_G_gt"] = Gg
        save_dict[f"{sp}_R_prev"] = Rp; save_dict[f"{sp}_R_pred"] = Rpr; save_dict[f"{sp}_R_gt"] = Rg
        save_dict[f"{sp}_delta_X"] = Xg - Xpr; save_dict[f"{sp}_delta_q"] = Xg[:, 7:28] - Xpr[:, 7:28]
        save_dict[f"{sp}_delta_K"] = Kg - Kpr; save_dict[f"{sp}_delta_G"] = Gg - Gpr
        save_dict[f"{sp}_traj_ids"] = tids

    np.savez_compressed(cache_file, **save_dict)
    print(f"[Dataset] Saved {traj_id} trajectories to {cache_file.name}")
    return splits
