"""Unit tests verifying persistent state, derived geometry, and relationship architecture."""

import sys
from pathlib import Path
import unittest
import importlib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.import_module("00_spec.state_definition")
_dyn = importlib.import_module("02_state.dynamics")
_fk = importlib.import_module("02_state.forward_kinematics")
_mano = importlib.import_module("02_state.mano_surface")
_rel = importlib.import_module("03_observation.relationship")

HandState = _spec.HandState
StateSnapshot = _spec.StateSnapshot
predict_kinematics = _dyn.predict_kinematics
predict_full_snapshot = _dyn.predict_full_snapshot
forward_kinematics = _fk.forward_kinematics
ManoSurface = _mano.ManoSurface
compute_skeleton_geometry = _rel.compute_skeleton_geometry
compute_glove_geometry = _rel.compute_glove_geometry
evaluate_relationship = _rel.evaluate_relationship
evaluate_consistency = _rel.evaluate_consistency

class TestStateArchitecture(unittest.TestCase):

    def setUp(self):
        self.state = HandState.neutral(z_depth=0.52)
        self.mano = ManoSurface()

    def test_state_vectorization_roundtrip(self):
        """Verify state serialization to continuous vector and reconstruction."""
        vec = self.state.to_vector()
        restored = HandState.from_vector(vec)
        np.testing.assert_allclose(restored.T, self.state.T)
        np.testing.assert_allclose(restored.R, self.state.R)
        np.testing.assert_allclose(restored.q, self.state.q)
        np.testing.assert_allclose(restored.beta, self.state.beta)
        np.testing.assert_allclose(restored.delta_G, self.state.delta_G)

    def test_relationship_recomputation_invariance(self):
        """Verify R_t is strictly deterministic and identical when recomputed from (K, G)."""
        if not self.mano.is_available:
            self.skipTest("MANO asset not loaded.")

        j_pos, j_rot = forward_kinematics(self.state)
        K = compute_skeleton_geometry(j_pos, j_rot)
        verts, faces, k21 = self.mano.forward(self.state, j_pos)
        G = compute_glove_geometry(verts, faces, k21)

        R1 = evaluate_relationship(K, G)
        R2 = evaluate_relationship(K, G)

        np.testing.assert_array_equal(R1.to_vector(), R2.to_vector())
        self.assertGreaterEqual(len(R1.to_vector()), 164)
        self.assertTrue(np.all(R1.bone_mean_thickness > 0.005))

    def test_temporal_prediction_flow(self):
        """Verify temporal snapshot transition evaluates R^- from (K^-, G^-)."""
        if not self.mano.is_available:
            self.skipTest("MANO asset not loaded.")

        self.state.V = np.array([0.1, -0.05, 0.02])
        self.state.qdot[6] = 0.5  # Index flexion rate

        j_pos, j_rot = forward_kinematics(self.state)
        K = compute_skeleton_geometry(j_pos, j_rot)
        verts, faces, k21 = self.mano.forward(self.state, j_pos)
        G = compute_glove_geometry(verts, faces, k21)
        R = evaluate_relationship(K, G)

        snapshot_prev = StateSnapshot(self.state, K, G, R, timestamp=0.0)
        snapshot_pred = predict_full_snapshot(snapshot_prev, self.mano, dt=0.016)

        # Confirm temporal progression
        self.assertGreater(np.linalg.norm(snapshot_pred.state.T - snapshot_prev.state.T), 1e-4)
        self.assertGreater(abs(snapshot_pred.state.q[6] - snapshot_prev.state.q[6]), 1e-4)

        # Consistency health validation
        score, pen, diag = evaluate_consistency(snapshot_pred.skeleton, snapshot_pred.glove)
        self.assertGreater(score, 0.50)
        self.assertTrue(diag["is_valid"])

if __name__ == '__main__':
    unittest.main()
