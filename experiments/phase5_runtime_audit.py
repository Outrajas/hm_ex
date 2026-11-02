"""Smoke audit for the recurrent state fusion and detector-free render input."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

network = importlib.import_module("04_network.architecture")
state_def = importlib.import_module("00_spec.state_definition")
coord = importlib.import_module("00_spec.coordinate_system")
runtime = importlib.import_module("08_runtime.tracker")
render_module = importlib.import_module("08_runtime.render_input")


def main() -> None:
    torch.manual_seed(7)
    initial = state_def.HandState.neutral(z_depth=0.55)
    initial.beta = np.linspace(-0.2, 0.2, 10, dtype=np.float64)
    model = network.TemporalStateCorrector(hidden_dim=64).cpu()
    tracker = runtime.PersistentHandTracker(model, initial)

    first = tracker.step(torch.zeros(5, 96, 96), dt_seconds=1.0 / 30.0)
    second = tracker.step(
        torch.zeros(5, 96, 96), dt_seconds=1.0 / 30.0, measurement_valid=False
    )
    assert np.allclose(first.state.beta, initial.beta)
    assert np.allclose(second.state.beta, initial.beta)
    assert second.uncertainty.shape == (27,)
    print(
        "State fusion: passed; "
        f"valid authority={first.authority:.3f}, "
        f"dropout authority={second.authority:.3f}, "
        f"uncertainty grew={bool(np.all(second.uncertainty >= first.uncertainty))}"
    )

    intrinsics = coord.CameraIntrinsics(
        fx=900.0, fy=900.0, cx=333.5, cy=512.0, width=667, height=1024
    )
    renderer = render_module.RenderedPriorInput(
        intrinsics,
        mano_model_path=ROOT / "models" / "MANO_RIGHT.pkl",
        output_size=128,
    )
    image = np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.uint8)
    rendered, _ = renderer(image, initial)
    assert rendered.shape == (5, 128, 128)
    assert torch.isfinite(rendered).all()
    shape_variant = initial.copy()
    shape_variant.beta[0] = 2.0
    mesh_zero, _, _ = renderer.mano.forward(initial, biometrics=renderer.biometrics)
    mesh_shape, _, _ = renderer.mano.forward(shape_variant, biometrics=renderer.biometrics)
    shape_displacement = float(np.linalg.norm(mesh_zero - mesh_shape, axis=-1).mean())
    assert shape_displacement > 1e-6
    print(
        f"Rendered input: shape={tuple(rendered.shape)}, "
        f"silhouette pixels={(rendered[3] > 0).sum().item()}, "
        f"skeleton pixels={(rendered[4] > 0).sum().item()}, "
        f"mean beta effect={shape_displacement:.6f} m"
    )


if __name__ == "__main__":
    main()
