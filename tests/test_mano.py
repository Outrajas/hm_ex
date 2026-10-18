"""Automated unit test suite for MANO integration and regression testing."""

import sys
from pathlib import Path
import unittest
import importlib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_state_def = importlib.import_module("00_spec.state_definition")
_mano = importlib.import_module("02_state.mano_surface")

HandState = _state_def.HandState
ManoSurface = _mano.ManoSurface
our_state_to_mano = _mano.our_state_to_mano

class TestManoIntegration(unittest.TestCase):

    def setUp(self):
        self.state = HandState.neutral(z_depth=0.52)
        self.mano = ManoSurface()

    def test_mano_module_import_and_availability(self):
        """Verify MANO module loads without errors and reports availability status."""
        self.assertIsInstance(self.mano.is_available, bool)

    def test_state_to_mano_mapping_dimensions(self):
        """Verify state conversion outputs correct tensor dimensions."""
        trans, root_aa, pose_aa, betas = our_state_to_mano(self.state)
        self.assertEqual(trans.shape, (3,))
        self.assertEqual(root_aa.shape, (3,))
        self.assertEqual(pose_aa.shape, (45,))
        self.assertEqual(betas.shape, (10,))
        self.assertFalse(np.isnan(pose_aa).any())

    def test_mano_forward_invariance_and_geometry(self):
        """Runs geometry assertions if the MANO model asset is available."""
        if not self.mano.is_available:
            self.skipTest("MANO_RIGHT.pkl asset not present in models/ - skipping forward geometry assertions.")

        verts, faces, joints = self.mano.forward(self.state)

        # Check shapes
        self.assertEqual(verts.shape, (778, 3))
        self.assertEqual(faces.shape, (1538, 3))
        self.assertEqual(joints.shape, (21, 3))

        # Check finiteness
        self.assertTrue(np.isfinite(verts).all(), "Vertices contain NaN or Inf values.")
        self.assertTrue(np.isfinite(joints).all(), "Joints contain NaN or Inf values.")

        # Test Translation Invariance
        delta_T = np.array([0.05, -0.02, 0.08])
        state_translated = self.state.copy()
        state_translated.T += delta_T
        verts_t, _, joints_t = self.mano.forward(state_translated)
        np.testing.assert_allclose(verts_t, verts + delta_T, atol=1e-7)
        np.testing.assert_allclose(joints_t, joints + delta_T, atol=1e-7)

        # Test Finger Articulation Responsiveness
        state_flex = self.state.copy()
        state_flex.q[6] = 1.4  # Index MCP flexion
        verts_flex, _, _ = self.mano.forward(state_flex)
        self.assertFalse(np.allclose(verts, verts_flex), "Finger flexion failed to deform MANO mesh.")

if __name__ == '__main__':
    unittest.main()
