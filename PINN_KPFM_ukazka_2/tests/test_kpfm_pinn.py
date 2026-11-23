import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kpfm_pinn import ROOT, Geometry, Measurement, load_all, load_measurement, validate_config


class DataAndGeometryTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((ROOT / "pinn_config.json").read_text())

    def test_real_exports_keep_first_row_and_time_units(self):
        measurements = load_all(ROOT / "data", self.c)
        self.assertEqual([m.rh for m in measurements], [5, 20, 40, 50, 60, 70, 80])
        for m in measurements:
            self.assertEqual(m.voltage.shape, (256, 7))
            self.assertEqual(m.x_um[0], 2.)
            np.testing.assert_array_equal(m.times_s, [0, 5, 10, 15, 20, 25, 30]
                                          if m.rh == 80 else [0, 30, 60, 150, 300, 450, 600])
        rh50 = next(m for m in measurements if m.rh == 50)
        self.assertAlmostEqual(rh50.voltage[0, 0], 1.469271104)

    def test_headerless_file_cannot_silently_guess_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "RH10.dat"
            p.write_text("0 1 2\n1 1 2\n2 1 2\n")
            with self.assertRaisesRegex(ValueError, "missing time header"):
                load_measurement(p, self.c)

    def test_bad_time_column_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "RH10.dat"
            p.write_text("x 0 s 2 s\n0 1\n1 1\n2 1\n")
            with self.assertRaisesRegex(ValueError, "voltage columns"):
                load_measurement(p, self.c)

    def test_geometry_samples_include_all_exposed_electrode_faces(self):
        g = Geometry(self.c)
        rng = np.random.default_rng(13)
        p = g.interior(rng, 3000)
        self.assertTrue(g.dielectric(p[:, :2]).all())
        self.assertTrue(((p > 0) & (p < 1)).all())
        left, right = g.electrodes(rng, 3000)
        for electrode, tip in ((left, .4), (right, .6)):
            on_edge = (np.isclose(electrode[:, 1], .4)
                       | np.isclose(electrode[:, 1], .6)
                       | np.isclose(electrode[:, 0], tip))
            self.assertTrue(on_edge.all())
            for mask in (np.isclose(electrode[:, 1], .4),
                         np.isclose(electrode[:, 1], .6),
                         np.isclose(electrode[:, 0], tip)):
                self.assertTrue(mask.any())
        outer, normal = g.outer(rng, 300)
        self.assertTrue(g.dielectric(outer[:, :2]).all())
        np.testing.assert_array_equal((normal**2).sum(1), np.ones(len(normal)))
        # Outward normals must agree with the coordinate of their boundary.
        np.testing.assert_allclose((outer[:, :2] * normal).sum(1),
                                   (normal > 0).sum(1))

    def test_geometry_validation(self):
        self.c["electrode_length_um"] = 6
        with self.assertRaises(ValueError):
            validate_config(self.c)


class PhysicsTests(unittest.TestCase):
    def test_independent_2d_temporal_inverse_recovers_known_D(self):
        from kpfm_reference import Reference
        c = json.loads((ROOT / "pinn_config.json").read_text())
        c["initial_condition"] = "extrude_first_profile"
        c["max_measurement_time_offset_s"] = 0.
        x = np.linspace(2, 8, 13)  # observation nodes on the reference grid
        s = np.clip((x-4)/2, 0, 1)
        u = 1-s + .3*np.sin(np.pi*s)
        initial = .72*(2*u)+.015
        times = np.array([0, 5, 30, 60, 150, 300, 600.])
        m = Measurement(Path("synthetic_RH50.dat"), 50, x, times,
                        np.repeat(initial[:, None], len(times), axis=1), "synthetic")
        generator = Reference(m, c, grid=21)
        truth = 2e-14
        fields = generator.fields(truth, .72, .015)
        np.testing.assert_allclose(fields["V_KPFM"][:, 10, 4:17].T,
                                   generator.predict(truth, .72, .015), atol=1e-12)
        np.testing.assert_allclose(fields["phi_V"][:, 10, :9], 2., atol=1e-12)
        np.testing.assert_allclose(fields["phi_V"][:, 10, 12:], 0., atol=1e-12)
        m.voltage = generator.predict(truth, .72, .015)
        fit = Reference(m, c, grid=21).fit(holdout=[3])
        self.assertLess(abs(fit["D_eff_m2_s"]/truth-1), .005)
        self.assertLess(fit["heldout_dielectric_RMSE_V"], 1e-5)

    def test_initial_state_is_only_left_electrode_and_short_delay_is_recovered(self):
        from kpfm_reference import Reference
        from kpfm_pinn import initial_envelope
        c = json.loads((ROOT / "pinn_config.json").read_text())
        c["initial_condition"] = "left_electrode_only"
        c["max_measurement_time_offset_s"] = 3.
        x = np.linspace(2, 8, 13)
        times = np.array([0., 5, 10, 30, 150, 600])
        m = Measurement(Path("synthetic_RH60.dat"), 60, x, times,
                        np.ones((len(x), len(times))), "synthetic")
        reference = Reference(m, c, grid=21)
        initial = reference.fields(2e-14, .72, .015, times_s=[0.])
        mask = initial["dielectric_mask"]
        np.testing.assert_allclose(initial["phi_V"][0][mask], 0., atol=1e-12)
        np.testing.assert_allclose(initial["phi_V"][:, 10, :9], 2., atol=1e-12)
        np.testing.assert_allclose(initial["phi_V"][:, 10, 12:], 0., atol=1e-12)
        np.testing.assert_array_equal(initial_envelope([[1., 0.], [1., 10.], [5., 5.]], c), [0., 0., 0.])
        m.voltage = reference.predict(2e-14, .72, .015, time_offset_s=1.8)
        fit = reference.fit(holdout=[3])
        self.assertLess(abs(fit["D_eff_m2_s"]/2e-14-1), .005)
        self.assertAlmostEqual(fit["measurement_time_offset_s"], 1.8, places=3)
        np.testing.assert_allclose(reference.fields(2e-14, .72, .015, time_offset_s=1.8)["V_KPFM"][:, 10, 4:17].T,
                                   m.voltage, atol=1e-12)

    def test_manufactured_diffusion_and_inverse_parameter_gradient(self):
        import torch
        from kpfm_pinn import derivatives
        alpha_true = .17

        class Analytic(torch.nn.Module):
            def __init__(self, alpha):
                super().__init__()
                self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(alpha), dtype=torch.float64))

            @property
            def alpha(self):
                return self.log_alpha.exp()

            def forward(self, p):
                # Exact solution u_t = alpha_true * (u_xx + u_yy).
                return (torch.cos(torch.pi*p[:, :1]) * torch.cos(torch.pi*p[:, 1:2])
                        * torch.exp(-2*torch.pi**2*alpha_true*p[:, 2:3]))

        torch.manual_seed(11)
        p = torch.rand(60, 3, dtype=torch.float64, requires_grad=True)
        exact = Analytic(alpha_true)
        self.assertLess(derivatives(exact, p).abs().max().item(), 1e-12)
        wrong = Analytic(.035)
        opt = torch.optim.Adam(wrong.parameters(), lr=.06)
        for _ in range(200):
            opt.zero_grad()
            residual = derivatives(wrong, p)
            loss = residual.square().mean()
            loss.backward()
            opt.step()
            p.grad = None
        self.assertLess(abs(wrong.alpha.item()/alpha_true-1), .002)

    def test_affine_voltage_and_diffusion_scaling(self):
        import torch
        from kpfm_pinn import make_model
        c = json.loads((ROOT / "pinn_config.json").read_text())
        model = make_model(c, 600, 8, 2, "cpu")
        D = model.alpha.item() * (10e-6)**2 / 600
        self.assertAlmostEqual(D/c["D_initial_m2_s"], 1, places=5)
        with torch.no_grad():
            model.log_gain.fill_(np.log(.7))
            model.offset.fill_(.02)
            observed = model.observed(torch.tensor([[0.], [1.]]))
        np.testing.assert_allclose(observed.numpy(), [[.02], [1.42]], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
