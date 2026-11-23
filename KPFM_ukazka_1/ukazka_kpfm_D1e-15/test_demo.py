"""Meaningful checks of the observation operator, PDE and blind split."""
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from physics import Surface, SpatialBasis
from pinn import ClockPINN
from run_demo import validate_config

ROOT = Path(__file__).resolve().parent


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
        cls.surface = Surface(cls.c["model"], 20)
        cls.basis = SpatialBasis(cls.surface)

    def test_psf_preserves_constant_and_mixes_across_metal_edge(self):
        H = self.surface.psf(np.array([[5, 5], [4, 5], [2, 5]]))
        np.testing.assert_allclose(H @ np.ones(len(self.surface.xy)), 1., atol=1e-14)
        self.assertGreater(H[1, self.surface.left].sum(), .4)
        self.assertLess(H[1, self.surface.left].sum(), .6)
        self.assertGreater(H[2, self.surface.left].sum(), .99)

    def test_generator_and_steady_state(self):
        A = self.surface.A
        symmetric = self.surface.mass[:, None]*A.toarray()
        np.testing.assert_allclose(symmetric, symmetric.T, rtol=1e-12, atol=1e-9)
        self.assertGreater(self.basis.rates.min(), 0.)
        fields = self.surface.simulate([0, 10, 200], self.c["diagnostics"]["D_nominal_m2_s"])
        self.assertGreaterEqual(fields.min(), -1e-12)
        self.assertLessEqual(fields.max(), 2+1e-12)
        np.testing.assert_array_equal(fields[:, self.surface.left], 2.)
        np.testing.assert_array_equal(fields[:, self.surface.right], 0.)

    def test_modal_and_independent_sparse_propagation_agree(self):
        points = np.array([[5., 5.], [3., 3.], [7., 6.5]])
        times = np.array([0., 3., 65., 600.])
        D = self.c["diagnostics"]["D_nominal_m2_s"]
        fields = self.surface.simulate(times, D)
        sparse = self.surface.psf(points) @ fields.T
        modal = self.basis.predict(points, times, D)
        np.testing.assert_allclose(sparse, modal, atol=1e-11)

    def test_clock_initial_condition_residual_and_parameter_gradient(self):
        net = ClockPINN(self.c["model"]).double()
        tau = torch.linspace(0, 1, 17, dtype=torch.float64)[:, None].requires_grad_(True)
        self.assertEqual(net.clock(tau)[0].item(), 0.)
        residual, _ = net.residual(tau)
        self.assertLess(residual.abs().max().item(), 1e-7)
        with torch.no_grad():
            net.log_alpha.add_(.5)
        residual, _ = net.residual(tau)
        residual.square().mean().backward()
        self.assertGreater(abs(net.log_alpha.grad.item()), 0.)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.net.parameters()))

    def test_invalid_scan_timing_rejected(self):
        c = json.loads(json.dumps(self.c))
        c["design"]["point_dwell_s"] = 1.
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_config(c)

    def test_saved_run_recovers_D_and_predicts_unseen_data(self):
        # Integration check uses real saved training output, never a mocked D.
        out = ROOT/"vystupy"
        if not (out/"fit_summary.json").exists():
            self.skipTest("Run run_demo.py all first for the integration check.")
        report = json.loads((out/"fit_summary.json").read_text(encoding="utf-8"))
        recovery = json.loads((out/"synthetic_recovery_check.json").read_text(encoding="utf-8"))
        self.assertLess(recovery["relative_error"], .1)
        self.assertLess(report["physics_clock_relative_RMS"], .05)
        self.assertLess(report["time_holdout_RMSE_V"], .03)
        self.assertLess(report["spatial_holdout_RMSE_V"], .03)
        import csv
        with open(out/"measurements.csv", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(not int(r["train"]) for r in rows if r["role"] == "spatial_validation" or int(r["time_holdout"])))
        self.assertTrue(all(float(r["t_s"]) >= float(r["nominal_time_s"]) for r in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
