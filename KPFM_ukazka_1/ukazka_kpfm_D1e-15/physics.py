"""Edge-refined finite volumes and exactly area-integrated Gaussian KPFM PSF.

At D=1e-15 m2/s the 600-second diffusion length sqrt(2Dt) is about 1.1 um.
A graded mesh also resolves the earlier thin layer near every metal face.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply, spsolve


class Surface:
    def __init__(self, config, n):
        self.c = dict(config)
        L = config["domain_um"]
        faces = np.linspace(0, L, n+1)
        if config.get("edge_refinement", False):
            minimum = config["edge_min_step_um"]
            offsets = np.array([0, 1, 2.5, 5, 10, 20, 40, 70])*minimum
            edges = [config["electrode_length_um"], L-config["electrode_length_um"],
                     (L-config["electrode_width_um"])/2, (L+config["electrode_width_um"])/2]
            extra = np.concatenate([edge+sign*offsets for edge in edges for sign in (-1, 1)])
            faces = np.unique(np.round(np.r_[faces, extra], 10))
            faces = faces[(faces >= 0) & (faces <= L)]
        self.faces = faces
        self.axis = (faces[1:]+faces[:-1])/2
        self.width = np.diff(faces)/L
        self.n = n = len(self.axis)
        for edge in (config["electrode_length_um"], L-config["electrode_length_um"],
                     (L-config["electrode_width_um"])/2, (L+config["electrode_width_um"])/2):
            if not np.any(np.isclose(faces, edge, atol=1e-10)):
                raise ValueError("Grid must align with all metal faces.")
        xx, yy = np.meshgrid(self.axis, self.axis)
        self.xy = np.column_stack((xx.ravel(), yy.ravel()))
        self.full_mass = np.outer(self.width, self.width).ravel()
        self.left, self.right = self.metal_masks(self.xy)
        self.metal = self.left | self.right
        self.keep = ~self.metal
        self.mass = self.full_mass[self.keep]
        self.fixed = np.where(self.left, config["left_voltage_V"], config["right_voltage_V"])
        ids = np.full(n*n, -1, dtype=int)
        ids[self.keep] = np.arange(self.keep.sum())
        rows, cols, values = [], [], []
        rhs = np.zeros(self.keep.sum())
        centers = self.axis/L
        for k in np.flatnonzero(self.keep):
            j, i = divmod(k, n)
            row, diag = ids[k], 0.
            for jj, ii, dim in ((j, i-1, 0), (j, i+1, 0), (j-1, i, 1), (j+1, i, 1)):
                if not (0 <= jj < n and 0 <= ii < n):
                    continue
                q = jj*n+ii
                width = self.width[i if dim == 0 else j]
                distance = abs(centers[ii]-centers[i]) if dim == 0 else abs(centers[jj]-centers[j])
                weight = 2/width**2 if self.metal[q] else 1/(width*distance)
                diag -= weight
                if self.metal[q]:
                    rhs[row] += weight*self.fixed[q]
                else:
                    rows.append(row); cols.append(ids[q]); values.append(weight)
            rows.append(row); cols.append(row); values.append(diag)
        self.A = coo_matrix((values, (rows, cols)), shape=(len(rhs), len(rhs))).tocsr()
        self.steady = spsolve(self.A, -rhs)
        self.initial = np.full(len(rhs), config["initial_dielectric_V"])
        self.full_steady = self.expand(self.steady)

    def metal_masks(self, xy):
        xy = np.asarray(xy)
        c = self.c
        band = np.abs(xy[:, 1]-c["domain_um"]/2) < c["electrode_width_um"]/2
        return (band & (xy[:, 0] < c["electrode_length_um"]),
                band & (xy[:, 0] > c["domain_um"]-c["electrode_length_um"]))

    def expand(self, dielectric):
        result = self.fixed.copy()
        result[self.keep] = dielectric
        return result

    def psf(self, points, sigma=None):
        sigma = self.c["psf_sigma_um"] if sigma is None else sigma
        if sigma < 0:
            raise ValueError("PSF sigma must be nonnegative.")
        points = np.asarray(points)
        if sigma == 0:
            distance2 = ((points[:, None, :]-self.xy[None, :, :])**2).sum(2)
            H = np.zeros_like(distance2)
            H[np.arange(len(points)), distance2.argmin(1)] = 1.
            return H
        # Exact Gaussian integral over each nonuniform rectangular cell.
        from scipy.special import ndtr
        px, py = points[:, 0, None], points[:, 1, None]
        wx = ndtr((self.faces[None, 1:]-px)/sigma)-ndtr((self.faces[None, :-1]-px)/sigma)
        wy = ndtr((self.faces[None, 1:]-py)/sigma)-ndtr((self.faces[None, :-1]-py)/sigma)
        H = (wy[:, :, None]*wx[:, None, :]).reshape(len(points), -1)
        H /= H.sum(1, keepdims=True)
        return H

    def simulate(self, times_s, D):
        times_s = np.asarray(times_s)
        if np.any(np.diff(times_s) < 0) or np.any(times_s < 0):
            raise ValueError("Simulation times must be sorted and nonnegative.")
        state = self.initial-self.steady
        rate = D/(self.c["domain_um"]*1e-6)**2
        previous, fields = 0., []
        for time in times_s:
            if time > previous:
                state = expm_multiply(self.A*(rate*(time-previous)), state)
            fields.append(self.expand(self.steady+state))
            previous = time
        return np.asarray(fields)


class SpatialBasis:
    def __init__(self, surface):
        self.surface = surface
        root_mass = np.sqrt(surface.mass)
        symmetric = root_mass[:, None]*surface.A.toarray()/root_mass[None, :]
        np.testing.assert_allclose(symmetric, symmetric.T, rtol=1e-12, atol=1e-7)
        eigenvalues, weighted_vectors = eigh(symmetric, check_finite=False, driver="evd")
        self.rates = -eigenvalues
        if self.rates.min() <= 0:
            raise ValueError("Diffusion generator must have negative eigenvalues.")
        self.vectors = weighted_vectors/root_mass[:, None]
        self.coefficients = weighted_vectors.T @ (root_mass*(surface.initial-surface.steady))

    def observation(self, points, sigma=None):
        H = self.surface.psf(points, sigma)
        return H @ self.surface.full_steady, (H[:, self.surface.keep] @ self.vectors)*self.coefficients

    def predict(self, points, times_s, D, sigma=None):
        steady, modes = self.observation(points, sigma)
        q = D*np.asarray(times_s)/(self.surface.c["domain_um"]*1e-6)**2
        return steady[:, None]+modes @ np.exp(-self.rates[:, None]*q[None, :])

    def fields(self, q):
        values = self.surface.steady+(np.exp(-np.outer(q, self.rates))*self.coefficients) @ self.vectors.T
        return np.array([self.surface.expand(row) for row in values])
