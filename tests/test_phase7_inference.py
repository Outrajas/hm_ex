"""Regression checks for residual application and manual MANO articulation."""
import importlib
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
api = runpy.run_path(str(ROOT / 'experiments/phase7_webcam_test.py'))
runtime = importlib.import_module('08_runtime.tracker')


class ConstantCorrection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.dx = 0.01

    def forward_step(self, *args, **kwargs):
        return SimpleNamespace(
            hidden=torch.zeros(1), authority=torch.ones(1),
            delta_T=torch.tensor([[self.dx, 0., 0.]]), delta_R=torch.zeros(1, 3),
            delta_q=torch.zeros(1, 21), log_sigma=torch.zeros(1, 27),
        )


class InferenceRegression(unittest.TestCase):
    def test_residual_does_not_double_count_velocity(self):
        state, _ = api['make_seed'](0.5)
        state.V[:] = [0.1, 0., 0.]
        model = ConstantCorrection()
        tracker = runtime.PersistentHandTracker(model, state)
        crop = torch.zeros(5, 128, 128)
        first = tracker.step(crop, dt_seconds=0.1).state
        np.testing.assert_allclose(first.T, state.T + [0.01, 0., 0.], atol=1e-7)
        model.dx = 0.
        second = tracker.step(crop, dt_seconds=0.1).state
        np.testing.assert_allclose(second.T, first.T, atol=1e-7)
        tracker.state.V[:] = [0.1, 0., 0.]
        missing = tracker.step(crop, dt_seconds=0.1, measurement_valid=False).state
        np.testing.assert_allclose(missing.T, second.T + [0.01, 0., 0.], atol=1e-7)

    def test_each_slider_articulates_its_skeleton_and_mesh(self):
        state, _ = api['make_seed'](0.5)
        bio = api['HandBiometrics']()
        mano = importlib.import_module('02_state.mano_surface').ManoSurface(str(ROOT / 'models/MANO_RIGHT.pkl'))
        joints, _ = api['forward_kinematics'](state, bio)
        vertices, _, _ = mano.forward(state, joints, bio)
        for finger, tip in enumerate((4, 8, 12, 16, 20)):
            curls = [0] * 5
            curls[finger] = 100
            bent = state.copy()
            bent.q = api['finger_curl_target'](state, curls)
            moved, _ = api['forward_kinematics'](bent, bio)
            mesh, _, _ = mano.forward(bent, moved, bio)
            self.assertGreater(np.linalg.norm(moved[tip] - joints[tip]), 0.01)
            self.assertGreater(np.max(np.linalg.norm(mesh - vertices, axis=1)), 0.01)
            opened = bent.copy()
            opened.q = api['finger_curl_target'](bent, [0] * 5)
            reopened, _ = api['forward_kinematics'](opened, bio)
            np.testing.assert_allclose(reopened, joints, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
