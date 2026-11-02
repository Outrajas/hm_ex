"""Automated unit tests for Stage-1 deterministic kinematics."""

import sys
from pathlib import Path
import unittest
import importlib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_coord = importlib.import_module("00_spec.coordinate_system")
_topo = importlib.import_module("00_spec.hand_topology")
_spec_const = importlib.import_module("00_spec.constraints")
_state_def = importlib.import_module("00_spec.state_definition")
_state_engine = importlib.import_module("02_state.state")
_state_const = importlib.import_module("02_state.constraints")
_fk = importlib.import_module("02_state.forward_kinematics")
_proj = importlib.import_module("03_observation.projection")

CameraIntrinsics = _coord.CameraIntrinsics
HandState = _state_def.HandState
ControlInput = _state_engine.ControlInput
state_update = _state_engine.state_update
project_constraints = _state_const.project_constraints
forward_kinematics = _fk.forward_kinematics
project_points_to_camera = _proj.project_points_to_camera
BONE_LENGTHS = _topo.BONE_LENGTHS

class TestStage1Decoupled(unittest.TestCase):

    def setUp(self):
        self.state = HandState.neutral(z_depth=0.5)
        self.cam = CameraIntrinsics(600.0, 600.0, 320.0, 240.0, 640, 480)

    def test_state_update_velocity(self):
        dt = 0.02
        target_T = self.state.T + np.array([0.04, -0.02, 0.06])
        control = ControlInput(target_T=target_T, authority=1.0)
        updated = state_update(self.state, control, dt)
        np.testing.assert_allclose(updated.T, target_T, atol=1e-10)
        np.testing.assert_allclose(updated.V, np.array([0.04, -0.02, 0.06]) / dt, atol=1e-10)

    def test_bone_length_invariance(self):
        self.state.q[5:9] = [0.2, 0.8, 0.6, 0.4]
        pos, _ = forward_kinematics(self.state)
        for (p, c), expected_len in BONE_LENGTHS.items():
            self.assertAlmostEqual(np.linalg.norm(pos[c] - pos[p]), expected_len, places=4)

    def test_projection_math(self):
        pt = np.array([[0.0, 0.0, 1.0]])
        uv, valid = project_points_to_camera(pt, self.cam)
        self.assertTrue(valid[0])
        self.assertAlmostEqual(uv[0, 0], 320.0)
        self.assertAlmostEqual(uv[0, 1], 240.0)

if __name__ == '__main__':
    unittest.main()
