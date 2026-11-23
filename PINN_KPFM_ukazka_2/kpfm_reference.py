"""Independent finite-difference inverse for checking the PINN estimate of D.

The spatial generator is diagonalized once. A scalar D and affine KPFM
calibration then fit the complete temporal profiles, with the same geometry,
initial-map closure and boundary conditions as the PINN. This is a numerical
reference, not a neural network, and does not resolve model identifiability.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class Reference:
    def __init__(self, measurement, config, grid=31):
        from kpfm_pinn import Geometry, initial_envelope
        if grid < 11 or grid > 81:
            raise ValueError("Reference grid must be between 11 and 81.")
        self.m, self.c, self.grid = measurement, config, grid
        g = Geometry(config)
        n = grid
        s = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(s, s, indexing="xy")
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        left, right = g.metal_masks(xy)
        metal = left | right
        keep = ~metal
        ids = np.full(n*n, -1, dtype=int)
        ids[keep] = np.arange(keep.sum())
        A = lil_matrix((keep.sum(), keep.sum()))
        rhs = np.zeros(keep.sum())
        known = np.where(left, config["left_voltage_V"], config["right_voltage_V"])
        h = 1/(n-1)
        for k in np.flatnonzero(keep):
            j, i = divmod(k, n)
            neighbours = [(j, i-1 if i else 1), (j, i+1 if i<n-1 else n-2),
                          (j-1 if j else 1, i), (j+1 if j<n-1 else n-2, i)]
            row = ids[k]
            A[row, row] = -4/h**2
            for jj, ii in neighbours:
                q = jj*n+ii
                if metal[q]:
                    rhs[row] += known[q]/h**2
                else:
                    A[row, ids[q]] += 1/h**2
        A = A.tocsr()
        steady = spsolve(A, -rhs)
        # Reflecting ghost nodes have a symmetric generator in trapezoidal
        # quadrature weights; diagonalize that similar symmetric matrix.
        w1 = np.ones(n)
        w1[[0, -1]] = .5
        mass = np.outer(w1, w1).ravel()[keep]
        root_mass = np.sqrt(mass)
        symmetric = root_mass[:, None] * A.toarray() / root_mass[None, :]
        if not np.allclose(symmetric, symmetric.T):
            raise RuntimeError("Internal error: nonsymmetric weighted Laplacian.")
        values, vectors = eigh(symmetric, check_finite=False, driver="evd")
        self.eigenvalues = np.minimum(values, 0)
        B = vectors / root_mass[:, None]

        def coefficients(v):
            return vectors.T @ (root_mass*v)

        scan = np.column_stack((measurement.x_um/g.L,
                                np.full(len(measurement.x_um), config["scan_y_um"]/g.L)))
        # Bilinear interpolation as a linear observation operator.
        ix = np.minimum((scan[:, 0]*(n-1)).astype(int), n-2)
        iy = np.minimum((scan[:, 1]*(n-1)).astype(int), n-2)
        fx, fy = scan[:, 0]*(n-1)-ix, scan[:, 1]*(n-1)-iy

        def observe(full):
            full = np.asarray(full)
            out = np.zeros((len(scan),) + full.shape[1:])
            for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                                   (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
                ww = weight if full.ndim == 1 else weight[:, None]
                out += ww * full[(iy+dy)*n+ix+dx]
            return out

        full_B = np.zeros((n*n, len(values)))
        full_B[keep] = B
        self.HB = observe(full_B)
        full_steady = known.copy()
        full_steady[keep] = steady
        self.full_B = full_B
        self.full_steady = full_steady
        self.x_um = xx * g.L
        self.y_um = yy * g.L
        self.dielectric_mask = keep.reshape(n, n)
        self.steady_scan = observe(full_steady)
        initial_line = np.interp(xy[keep, 0]*g.L, measurement.x_um, measurement.voltage[:, 0])
        envelope = initial_envelope(xy[keep]*g.L, config)
        self.coeff_raw = coefficients(envelope*initial_line)
        self.coeff_gain = coefficients((1-envelope)*config["backgate_voltage_V"]-steady)
        self.coeff_offset = coefficients(-envelope)
        self.left, self.right = g.metal_masks(scan)
        self.L_m = g.L*1e-6

    def predict(self, D, gain, offset, time_offset_s=0.):
        exponential = np.exp(self.eigenvalues[:, None] * D/self.L_m**2 * (self.m.times_s[None, :]+time_offset_s))
        coefficients = self.coeff_raw + gain*self.coeff_gain + offset*self.coeff_offset
        result = gain*self.steady_scan[:, None] + offset + self.HB @ (coefficients[:, None]*exponential)
        result[self.left, :] = gain*self.c["left_voltage_V"]+offset
        result[self.right, :] = gain*self.c["right_voltage_V"]+offset
        return result

    def fields(self, D, gain, offset, times_s=None, time_offset_s=0.):
        """Return the same fitted numerical solution over the whole 2D grid.

        Arrays have shape (time, y, x). This is a conditional reconstruction
        using the configured initial-map closure, not a measured 2D map.
        """
        times = self.m.times_s if times_s is None else np.asarray(times_s, dtype=float)
        if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError("Field times must be a finite nonnegative vector.")
        if D <= 0 or gain <= 0 or not np.isfinite([D, gain, offset]).all():
            raise ValueError("D and gain must be positive and all parameters finite.")
        if not np.isfinite(time_offset_s) or time_offset_s < 0:
            raise ValueError("Time offset must be nonnegative and finite.")
        exponential = np.exp(self.eigenvalues[:, None] * D/self.L_m**2 * (times[None, :]+time_offset_s))
        coefficients = self.coeff_raw + gain*self.coeff_gain + offset*self.coeff_offset
        raw = (gain*self.full_steady[:, None] + offset
               + self.full_B @ (coefficients[:, None]*exponential))
        raw = raw.T.reshape(len(times), self.grid, self.grid)
        return {"x_um": self.x_um, "y_um": self.y_um, "times_s": times,
                "simulation_times_s": times+time_offset_s,
                "V_KPFM": raw, "phi_V": (raw-offset)/gain,
                "dielectric_mask": self.dielectric_mask}

    def fit(self, holdout=()):
        m, c = self.m, self.c
        if any(i <= 0 or i >= len(m.times_s) for i in holdout):
            raise ValueError("Holdout indices must refer to nonzero time columns.")
        columns = ~np.isin(np.arange(len(m.times_s)), list(holdout))
        fit_calibration = c["observation"] == "affine"
        max_delay = c.get("max_measurement_time_offset_s", 0.)
        fit_delay = max_delay > 0
        if fit_calibration:
            if not self.left.any() or not self.right.any():
                raise ValueError("Reference calibration needs both electrode plateaus.")
            gain0 = ((np.median(m.voltage[self.left][:, columns])
                      - np.median(m.voltage[self.right][:, columns]))
                     / (c["left_voltage_V"]-c["right_voltage_V"]))
            gain0 = max(gain0, .01)
            offset0 = np.median(m.voltage[self.right][:, columns])-gain0*c["right_voltage_V"]

        def decode(theta):
            base = (10.**theta[0], math.exp(theta[1]), theta[2]) if fit_calibration else (10.**theta[0], 1., 0.)
            delay = theta[-1]*max_delay if fit_delay else c.get("measurement_time_offset_s", 0.)
            return (*base, delay)

        def residual(theta):
            return (self.predict(*decode(theta))[:, columns]-m.voltage[:, columns]).ravel()

        best = None
        # Multiple starts explicitly check the scalar inverse, rather than
        # accepting the user's small-D expectation as evidence.
        for logD in (-18., -16., -14., -12.):
            initial = [logD, math.log(gain0), offset0] if fit_calibration else [logD]
            lo, hi = ([-22, -5, -5], [-10, 5, 5]) if fit_calibration else ([-22], [-10])
            if fit_delay:
                initial.append(.5)
                lo.append(0.)
                hi.append(1.)
            fit = least_squares(residual, initial, bounds=(lo, hi), max_nfev=100,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun):
                best = fit
        D, gain, offset, delay = decode(best.x)
        pred = self.predict(D, gain, offset, delay)
        gap = ~(self.left | self.right)
        held = ~columns
        return {"method": "finite_difference_temporal_inverse", "grid": self.grid,
                "D_eff_m2_s": D, "gain": gain, "offset_V": offset,
                "measurement_time_offset_s": delay,
                "time_offset_at_upper_bound": bool(fit_delay and delay > .99*max_delay),
                "fit_warning": "time_offset_at_upper_bound_check_initial_electrostatic_response" if fit_delay and delay > .99*max_delay else "",
                "first_profile_RMSE_V": float(np.sqrt(np.mean((pred[:, 0]-m.voltage[:, 0])**2))),
                "fit_RMSE_V": float(np.sqrt(np.mean((pred[:, columns]-m.voltage[:, columns])**2))),
                "heldout_dielectric_RMSE_V": float(np.sqrt(np.mean((pred[gap][:, held]-m.voltage[gap][:, held])**2))) if held.any() else None,
                "D_near_search_bound": bool(best.x[0] < -21.9 or best.x[0] > -10.1),
                "optimizer_success": bool(best.success),
                "conditional_on_same_physical_assumptions_as_PINN": True}
